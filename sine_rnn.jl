using Flux
using Flux: @epochs
using PyPlot
using Random
Random.seed!(1234)

input_size, hidden_size, output_size = 7, 6, 1
epochs = 600
seq_length = 200
lr = 0.1

data_time_steps = range(2, stop = 200, length = seq_length + 1)
timedata = sin.(data_time_steps)

x = timedata[1:end-1]
y = timedata[2:end]
model = Chain(RNN(1,6),Dense(6,1))
function eval_model(x)
  out = model(x)[end]
  #Flux.reset!(model)
  out
end
opt = ADAM(0.0005)

loss(x, y) = Flux.mse(eval_model(x), y)

ps = Flux.params(model)
evalcb() = @show(loss(x, y))
Flux.train!(loss, ps, zip(x,y), opt)
evalcb() = @show(sum(loss.(x, y)))
@epochs epochs Flux.train!(loss, ps, zip(x, y), opt, cb = Flux.throttle(evalcb, 1))

pred = Float32[]
for i = 1:length(x)
    push!(pred, model(x[i])[end])
end
figure()
plot(data_time_steps[2:end],pred)
plot(data_time_steps[2:end],y)

gcf()
