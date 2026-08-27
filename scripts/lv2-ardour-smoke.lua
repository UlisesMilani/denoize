-- Headless Ardour LV2 lifecycle smoke test.

local phase = assert(arg[1], "phase argument is required")
local session_directory = assert(arg[2], "session directory argument is required")
local session_name = "denoize-lv2-smoke"
local sample_rate = 48000
io.stdout:setvbuf("no")
assert(phase == "create" or phase == "restore", "phase must be create or restore")

local function progress(stage)
    print("DENOIZE_LV2_ARDOUR_PROGRESS stage=" .. stage)
end

local function require_pointer(value, label)
    assert(value ~= nil, label .. " is nil")
    if value.isnil ~= nil then
        assert(not value:isnil(), label .. " is an empty Ardour pointer")
    end
    return value
end

local function add_plugin(track, unique_id, expected_name)
    local processor = require_pointer(
        ARDOUR.LuaAPI.new_plugin(Session, unique_id, ARDOUR.PluginType.LV2, ""),
        expected_name .. " processor"
    )
    assert(processor:display_name() == expected_name, expected_name .. " descriptor mismatch")
    assert(track:add_processor_by_index(processor, 0, nil, true) == 0,
        "could not insert " .. expected_name)
    assert(processor:active(), expected_name .. " did not activate")
    return processor
end

local function process_transport_pass()
    local before = Session:transport_sample()
    Session:request_roll(ARDOUR.TransportRequestSource.TRS_UI)
    sleep(2)
    local after = Session:transport_sample()
    Session:request_stop(false, false, ARDOUR.TransportRequestSource.TRS_UI)
    sleep(1)
    assert(after > before, "Ardour transport did not process an audio block")
    return after - before
end

if phase == "create" then
    local created = create_session(session_directory, session_name, sample_rate)
    require_pointer(created, "created session")
    assert(Session:name() == session_name, "created session name mismatch")
    progress("session-created")

    local tracks = Session:new_audio_track(
        2,
        2,
        nil,
        2,
        "denoize-lv2",
        ARDOUR.PresentationInfo.max_order,
        ARDOUR.TrackMode.Normal,
        true
    )
    local standard_track = require_pointer(tracks:front(), "standard track")
    local neural_track = require_pointer(tracks:back(), "neural track")
    local standard = add_plugin(
        standard_track,
        "https://github.com/penguin425/denoize#lv2-dsp",
        "denoize"
    )
    local neural = add_plugin(
        neural_track,
        "https://github.com/penguin425/denoize#lv2-neural",
        "denoize Neural"
    )
    progress("plugins-inserted")
    local processed_frames = process_transport_pass()
    progress("create-pass-processed")
    assert(standard:signal_latency() == 480, "standard latency contract mismatch")
    assert(neural:signal_latency() == 11520, "neural latency contract mismatch")

    standard:deactivate()
    neural:deactivate()
    assert(not standard:active() and not neural:active(), "deactivation failed")
    standard:activate()
    neural:activate()
    assert(standard:active() and neural:active(), "reactivation failed")
    assert(Session:save_state("") == 0, "session state save failed")
    progress("state-saved")
    print(string.format(
        "DENOIZE_LV2_ARDOUR_CREATE processed_frames=%d sample_rate_hz=%d descriptors=2 state_saved=true",
        processed_frames,
        sample_rate
    ))
    print("DENOIZE_LV2_ARDOUR_LATENCY standard_frames=480 neural_frames=11520")
    close_session()
    assert(Session == nil, "created session did not close")
    progress("created-session-closed")
    print("DENOIZE_LV2_ARDOUR_TEARDOWN phase=create passed=true")
    return
end

local loaded = load_session(session_directory, session_name)
require_pointer(loaded, "restored session")
assert(Session:name() == session_name, "restored session name mismatch")
local standard_track = require_pointer(Session:route_by_name("denoize-lv2 1"), "standard track")
local neural_track = require_pointer(Session:route_by_name("denoize-lv2 2"), "neural track")
local standard = require_pointer(standard_track:nth_plugin(0), "standard processor")
local neural = require_pointer(neural_track:nth_plugin(0), "neural processor")
assert(standard:display_name() == "denoize", "standard state restore mismatch")
assert(neural:display_name() == "denoize Neural", "neural state restore mismatch")
assert(standard:active() and neural:active(), "restored processors are inactive")
progress("state-restored")
local processed_frames = process_transport_pass()
progress("restore-pass-processed")
assert(standard:signal_latency() == 480, "restored standard latency mismatch")
assert(neural:signal_latency() == 11520, "restored neural latency mismatch")
print(string.format(
    "DENOIZE_LV2_ARDOUR_RESTORE processed_frames=%d sample_rate_hz=%d descriptors=2 state_reload=true",
    processed_frames,
    sample_rate
))
print("DENOIZE_LV2_ARDOUR_LATENCY standard_frames=480 neural_frames=11520")
close_session()
assert(Session == nil, "restored session did not close")
progress("restored-session-closed")
print("DENOIZE_LV2_ARDOUR_TEARDOWN phase=restore passed=true")
