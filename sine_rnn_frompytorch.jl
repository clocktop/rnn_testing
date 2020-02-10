using Flux
#using Flux.Tracker
using PyPlot
using Random
Random.seed!(1234)

input_size, hidden_size, output_size = 7, 6, 1
epochs = 200
seq_length = 20
lr = 0.1

data_time_steps = range(2, stop = 10, length = seq_length + 1)
timedata = sin.(data_time_steps)

x = timedata[1:end-1]
y = timedata[2:end]

w1 = param(randn(input_size, hidden_size))
w2 = param(randn(hidden_size, output_size))

function forward(input, context_state, W1, W2)
    xh = cat(dims=2, input, context_state)
    #xh = cat(dims=1, Tracker.collect(input), context_state)
    context_state = tanh.(xh * W1)
    out = context_state*W2
    return out, context_state
end

function train()
    for i = 1:epochs
        total_loss = 0
        context_state = param(zeros(1, hidden_size))
        for j = 1:length(x)
            input = x[j]
            target = y[j]
            pred, context_state = forward(input, context_state, w1, w2)
            loss = param(sum((pred .- target) .^ 2) / 2)
            total_loss += loss
            Tracker.back(loss)
            w1.data .-= lr .* w1.grad
            w2.data .-= lr .* w2.grad
            w1.grad .= 0.0
            w2.grad .= 0.0
            context_state = param(context_state.data)
        end
        if (i % 10 == 0)
            println("Epoch: $i  loss: $total_loss")
        end
    end
end

#train()

context_state = param(zeros(1, hidden_size))
predictions = []

for i = 1:length(x)
    input = x[i]
    pred, context_state = forward(input, context_state, w1, w2)
    append!(predictions, pred.data)
end
scatter(data_time_steps[1:end-1], x, label = "actual")
scatter(data_time_steps[2:end], predictions, label = "predicted")
gcf()
