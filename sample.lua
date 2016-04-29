require 'load_midi'
require 'rbm'
require 'nn'
require 'rnn'
require 'gnuplot'

MIDI=require 'MIDI'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from the NN-RBM music generator model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','train','data directory. Should contain MIDI files.')
-- model params
cmd:option('-rnn_model','models/recurrence-rnn_10.dat','Recurrent module')
cmd:option('-mlp_model','models/recurrence-mlp_10.dat','MLP module with bias and RBM part')

cmd:option('-n_hidden', 20, 'RBM hidden layer size.')
cmd:option('-n_recurrent', 50, 'Recurrent hidden size.')

cmd:option('-length',300,'sample length')
cmd:option('-rho',32,'number of timesteps to unroll for')
cmd:option('-o', 'sampled.mid', 'output mid file')

opt = cmd:parse(arg)
torch.seed()

opt.n_visible = roll_height

init_pool(opt)

rnn = torch.load(opt.rnn_model)
mlp = torch.load(opt.mlp_model)

rnn:forget()
mlp:forget()
rnn:evaluate()
mlp:evaluate()

piano_roll = torch.Tensor(opt.length, opt.n_visible)
zeros = torch.zeros(opt.n_visible)

rnn_outputs={}
rnn_outputs[0] = torch.zeros(opt.n_recurrent)

-- initial sequence
for t=1, opt.rho do
    rnn_outputs[t] = rnn:forward(input_pool[t+256]:double())
    local sampled_v = mlp:forward{input_pool[t+256]:double(), rnn_outputs[t-1]}

    piano_roll[t]:copy(sampled_v)
end

-- sample whole piano-roll
for t=opt.rho+1, opt.length do
	rnn_outputs[t] = rnn:forward(input_pool[t+256+opt.rho]:double())
    local sampled_v = mlp:forward{input_pool[t+256+opt.rho]:double(), rnn_outputs[t-1]}
    piano_roll[t]:copy(sampled_v)
end

piano_roll = piano_roll[{{opt.rho, opt.length}, {}}]
opt.length = opt.length-opt.rho-1

gnuplot.pngfigure(opt.o..'.png')
gnuplot.imagesc(piano_roll:t())
gnuplot.plotflush()

ticks = 96
division_4 = ticks
division_32 = division_4 / 4

score = {ticks, {}}
score[2][1] = {"patch_change", 0, 0, 0}
score[2][2] = {'set_tempo', 0, 1200000}

counter = 3
for t=1, opt.length do
    for note=1, roll_height do
        if piano_roll[t][note] == 1 then
            -- trace note and erase
            local trace = t
            while trace < opt.length-1 and (piano_roll[trace+1][note] == 1 or piano_roll[trace+2][note] == 1) do
                trace = trace + 1
            end
            local duration = trace - t + 1
            piano_roll[{{t, trace}, note}]:fill(0)
            
            target_note = note + 20
            if target_note >= 0 and target_note <= 127 then
                score[2][counter] = {"note", (t-1)*division_32, duration*division_32, 0, target_note, 90}
                counter = counter + 1
            end
        end
    end
end

midifile = assert(io.open(opt.o,'w'))
midifile:write(MIDI.score2midi(score))
midifile:close()
