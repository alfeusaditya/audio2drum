-- Drum audio to MIDI (Addictive Drums 2)
-- Select a drum audio item, run this action. It analyses the audio with audio2drum.py,
-- (optionally) writes a tempo map that follows the drummer, and adds a new track with
-- a MIDI item (AD2 note map) snapped to that grid, lined up with the audio. No VSTi is loaded.
-- Stems (ISO Drums "<name>_kick.wav" ... or LarsNet kick.wav ...) next to the audio file
-- or in a sub-folder are used automatically.

local dir = ({ reaper.get_action_context() })[2]:match("^(.*)[/\\]")
local PY = dir .. "/venv/bin/python"
local SCRIPT = dir .. "/audio2drum.py"
local TITLE = "Drum to MIDI (AD2)"

local function fail(msg) reaper.MB(msg, TITLE, 0) end

local item = reaper.GetSelectedMediaItem(0, 0)
if not item then return fail("Pilih dulu item audio drum-nya.") end
local take = reaper.GetActiveTake(item)
if not take or reaper.TakeIsMIDI(take) then return fail("Item yang dipilih harus audio, bukan MIDI.") end
local src = reaper.GetMediaItemTake_Source(take)
local parent = reaper.GetMediaSourceParent(src)
if parent then src = parent end
local path = reaper.GetMediaSourceFileName(src, "")
if path == "" then return fail("File audio item ini tidak ditemukan.") end

local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
local len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
local offs = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
local rate = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
local function proj_time(t) return pos + (t - offs) / rate end -- seconds in the audio file -> project time

local ok, csv = reaper.GetUserInputs(TITLE, 5,
  "Birama (4/4, 6/8, ...),Downbeat detik di item (kosong=auto),BPM tetap (kosong=ikut drummer),Atur tempo project (y/n),Quantize 1/16 (y/n; n=tepat di transient),extrawidth=80",
  "4/4,,,n,n")
if not ok then return end
local f = {}
for v in (csv .. ","):gmatch("([^,]*),") do f[#f + 1] = v:match("^%s*(.-)%s*$") end
-- only an explicit "y" turns these on; anything else (n, empty) = off
local meter, anchor, bpm = f[1], f[2], f[3]
local set_tempo, quantize = (f[4] or ""):lower() == "y", (f[5] or ""):lower() == "y"
if not meter:match("^%d+/%d+$") then return fail("Birama harus seperti 4/4 atau 6/8.") end

local tmp = dir .. "/last"
local cmd = string.format('"%s" "%s" "%s" "%s.mid" --meter %s --notes "%s.txt"', PY, SCRIPT, path, tmp, meter, tmp)
if not quantize then cmd = cmd .. " --no-quantize" end
if anchor ~= "" then
  local a = tonumber(anchor); if not a then return fail("Downbeat harus angka detik.") end
  cmd = cmd .. string.format(" --anchor %.4f", offs + a * rate)
end
if bpm ~= "" then
  local b = tonumber(bpm); if not b then return fail("BPM harus angka.") end
  cmd = cmd .. string.format(" --bpm %.3f", b)
end
os.remove(tmp .. ".txt")
local out = reaper.ExecProcess(cmd, 0) or ""
local code, log = out:match("^(%-?%d+)\n?(.*)$")
local fh = io.open(tmp .. ".txt", "r")
if code ~= "0" or not fh then
  return fail("Analisis gagal:\n\n" .. (log or out):sub(-1500))
end

local num, den, beats, notes = 4, 4, {}, {}
for line in fh:lines() do
  local k, a, b, c = line:match("^(%u+)%s+(%S+)%s*(%S*)%s*(%S*)")
  if k == "METER" then num, den = tonumber(a), tonumber(b)
  elseif k == "BEAT" then beats[#beats + 1] = { proj_time(tonumber(a)), tonumber(b) }
  elseif k == "NOTE" then notes[#notes + 1] = { proj_time(tonumber(a)), tonumber(b), tonumber(c) }
  end
end
fh:close()
-- keep only notes inside the (possibly trimmed) item
local kept = {}
for _, n in ipairs(notes) do
  if n[1] >= pos - 0.05 and n[1] <= pos + len then kept[#kept + 1] = n end
end
if #kept == 0 then return fail("Tidak ada pukulan drum yang terdeteksi di item ini.") end

reaper.Undo_BeginBlock()
reaper.PreventUIRefresh(1)

if set_tempo then
  -- audio stays put in time while the tempo map changes around it
  reaper.SetMediaItemInfo_Value(item, "C_BEATATTACHMODE", 0)
  local qpb = num * 4 / den -- quarter beats per bar (BEAT lines start at bar 1)
  local first
  for i, b in ipairs(beats) do
    if b[1] >= 0 and (i - 1) % qpb == 0 then first = i break end
  end
  local stop = pos + len
  if first then
    -- REAPER snaps a time-signature marker to a bar line, so first make a lead-in bar(s)
    -- from the current bar line up to our bar 1, with whatever tempo makes them end exactly there.
    local T, bar_len = beats[first][1], qpb * 60 / beats[first][2]
    local _, meas = reaper.TimeMap2_timeToBeats(0, T)
    local L = reaper.TimeMap_GetMeasureInfo(0, meas)
    if T - L > 0.01 then
      local n = math.max(1, math.floor((T - L) / bar_len + 0.5))
      reaper.SetTempoTimeSigMarker(0, -1, L, -1, -1, n * qpb * 60 / (T - L), num, den, false)
    end
  end
  local last_bpm
  for i = first or #beats + 1, #beats do
    local b = beats[i]
    if b[1] > stop then break end
    if i == first then
      reaper.SetTempoTimeSigMarker(0, -1, b[1], -1, -1, b[2], num, den, false)
    elseif math.abs(b[2] - last_bpm) > 0.01 then
      reaper.SetTempoTimeSigMarker(0, -1, b[1], -1, -1, b[2], 0, 0, false)
    end
    last_bpm = b[2]
  end
  reaper.UpdateTimeline()
end

local track = reaper.GetMediaItem_Track(item)
local idx = math.floor(reaper.GetMediaTrackInfo_Value(track, "IP_TRACKNUMBER"))
reaper.InsertTrackAtIndex(idx, true)
local tr = reaper.GetTrack(0, idx)
reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "Drums MIDI (AD2)", true)

local t0 = math.max(0, math.min(pos, kept[1][1]))
local mi = reaper.CreateNewMIDIItemInProj(tr, t0, pos + len, false)
local mt = reaper.GetActiveTake(mi)
reaper.GetSetMediaItemTakeInfo_String(mt, "P_NAME", "Drums MIDI", true)
for _, n in ipairs(kept) do
  local p0 = reaper.MIDI_GetPPQPosFromProjTime(mt, n[1])
  local p1 = reaper.MIDI_GetPPQPosFromProjTime(mt, n[1] + 0.06)
  reaper.MIDI_InsertNote(mt, false, false, p0, p1, 9, n[2], n[3], true)
end
reaper.MIDI_Sort(mt)

reaper.PreventUIRefresh(-1)
reaper.UpdateArrange()
reaper.Undo_EndBlock("Drum audio to MIDI (AD2)", -1)

-- no console window: the new track/item is the result (errors still show a message box)
