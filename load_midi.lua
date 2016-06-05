-- load_midi.lua
-- Zdeněk Janeček, 2016 (ycdmdj@gmail.com)
--
-- University of West Bohemia

MIDI=require 'MIDI'
require 'paths'

input_pool = {}

roll_height = 88
number_count = 0

function init_pool(conf)
	for fname in paths.files(conf.data_dir) do
		if fname ~= '..' and fname ~= '.' then
			local piano_roll = load_song(conf.data_dir..'/'..fname)

			for t=1, piano_roll:size(2) do
				-- input
				local input = torch.ByteTensor(roll_height)
				input:copy(piano_roll[{{}, t}])
				table.insert(input_pool, input)

				if #input_pool%(conf.batch_size+conf.rho-1)==0 then
					number_count = number_count + 1
				end
			end
		end
	end
end

function load_batch(number, batch, conf)
    assert(number<=number_count, "non existing batch")

    -- because indexing from 1 sucks
    local begin = (number-1)*#batch+1

    for t=0, #batch-1 do
        for i=0, conf.batch_size-1 do
            batch[t+1][i+1]:copy(input_pool[begin+t+i])
        end
    end
end

function load_song(name)
    local f=assert(io.open(name, 'r'))
    local score=MIDI.midi2score(f:read('*all'))
    local stats=MIDI.score2stats(score)
    f:close()

    -- read only assigned tracks
    local assigned = {}
    for key, val in pairs(stats['channels_by_track']) do
        if #val > 0 then
            table.insert(assigned, key+1)
        end
    end

    local division_4 = score[1]
    local division_32 = division_4 / 4
    local song_len = stats['nticks']

    local song_bitmap = torch.ByteTensor(roll_height, math.ceil(song_len / division_32)+2):zero()

    -- for each track with music
    for key, track in pairs(assigned) do
        -- for each event
        for k,event in pairs(score[track]) do
            -- if it is note, not drum on channel 10 and regular instrument
            if event[1] == 'note' and event[4] ~= 10 and event[5] < 112 then
                start = math.ceil(event[2] / division_32) + 1
                duration = math.ceil(event[3] / division_32)
                note = event[5]

                if note >= 21 and note <= 108 then
					song_bitmap[{note-20, {start, start+duration}}] = 1
                end
            end
        end
    end

    return song_bitmap
end
