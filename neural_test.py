from neural import NeuralNet

print("\n\nTraining RN\n\n")
rn_training_data = [
    ([1, 0, 1, 0, 0, 0], [1]),
    ([1, 0, 1, 1, 0, 0], [1]),
    ([1, 0, 1, 0, 1, 0], [1]),
    ([1, 1, 0, 0, 1, 1], [1]),
    ([1, 1, 1, 1, 0, 0], [1]),
    ([1, 0, 0, 0, 1, 1], [1]),
    ([1, 0, 0, 0, 1, 0], [0]),
    ([0, 1, 1, 1, 0, 1], [1]),
    ([0, 1, 1, 0, 1, 1], [0]),
    ([0, 0, 0, 1, 1, 0], [0]),
    ([0, 1, 0, 1, 0, 1], [0]),
    ([0, 0, 0, 1, 0, 1], [0]),
    ([0, 1, 1, 0, 1, 1], [0]),
    ([0, 1, 1, 1, 0, 0], [0]),
]

nn = NeuralNet(6, 1, 1)
nn.train(rn_training_data)

print(f"weights:{nn.get_ih_weights()}")
print()
print(f"Weights:{nn.get_ho_weights()}")

print(nn.evaluate([1, 1, 1, 1, 1, 1]))
print(nn.evaluate([0, 0, 0, 0, 0, 0]))
print()
print(nn.evaluate([1, 0, 0, 0, 0, 0]))  # 1
print(nn.evaluate([0, 1, 0, 0, 0, 0]))  # 0
print(nn.evaluate([0, 0, 1, 0, 0, 0]))  # 0
print(nn.evaluate([0, 0, 0, 1, 0, 0]))  # 0
print(nn.evaluate([0, 0, 0, 0, 1, 0]))  # 0
print(nn.evaluate([0, 0, 0, 0, 0, 1]))  # 0
print()
print(nn.evaluate([0, 1, 1, 1, 1, 1]))  # 0
print(nn.evaluate([1, 0, 1, 1, 1, 1]))  # 1
print(nn.evaluate([1, 1, 0, 1, 1, 1]))  # 1
print(nn.evaluate([1, 1, 1, 0, 1, 1]))  # 1
print(nn.evaluate([1, 1, 1, 1, 0, 1]))  # 1
print(nn.evaluate([1, 1, 1, 1, 1, 0]))  # 1
print()
print(nn.evaluate([0, 1, 1, 1, 0, 1]))
print(nn.evaluate([1, 0, 0, 0, 1, 0]))

print("\n\nTraining GT\n\n")
gt_training_data = [
    ([1, 1, 0, 0], [1]),
    ([1, 1, 0, 1], [1]),
    ([1, 1, 1, 0], [1]),
    ([1, 0, 0, 0], [1]),
    ([1, 0, 0, 1], [1]),
    ([0, 1, 0, 0], [1]),
    ([1, 0, 1, 1], [0]),
    ([0, 1, 0, 0], [0]),
    ([0, 0, 0, 1], [0]),
    ([0, 0, 1, 1], [0]),
    ([0, 0, 1, 0], [0]),
    ([0, 0, 0, 1], [0]),
]

gtn = NeuralNet(4, 3, 1)
gtn.train(gt_training_data)

print(gtn.evaluate([0, 0, 1, 1]))
print(gtn.evaluate([0, 0, 0, 0]))
print(gtn.evaluate([1, 1, 1, 1]))
print(gtn.evaluate([1, 0, 1, 0]))
print(gtn.evaluate([0, 1, 0, 1]))

print("\n\nTraining IGT\n\n")
igt_training_data = [
    ([0.25, 0.00], [1.0]),
    ([0.50, 0.00], [1.0]),
    ([0.50, 0.25], [1.0]),
    ([0.75, 0.00], [1.0]),
    ([0.75, 0.25], [1.0]),
    ([0.75, 0.00], [1.0]),
    ([0.00, 0.25], [0.0]),
    ([0.00, 0.50], [0.0]),
    ([0.00, 0.75], [0.0]),
    ([0.25, 0.50], [0.0]),
    ([0.25, 0.75], [0.0]),
    ([0.50, 0.75], [0.0]),
]

igtn = NeuralNet(2, 3, 1)
igtn.train(igt_training_data)

print()
print(igtn.test_with_expected(igt_training_data))
print(igtn.evaluate([0.1, 0.5]))
print(igtn.evaluate([0.7, 0.1]))
print(igtn.evaluate([0.5, 0.5]))
print(igtn.evaluate([0.3, 0.3]))

print("\n\nTraining SQ\n\n")
sq_training_data = [
    ([0.2], [0.04]),
    ([0.3], [0.09]),
    ([0.5], [0.25]),
    ([0.7], [0.49]),
    ([0.1], [0.01]),
]
sqn = NeuralNet(1, 6, 1)
sqn.train(sq_training_data)

print()
print(sqn.test_with_expected(sq_training_data))
print(sqn.evaluate([0.66]))
print(sqn.evaluate([0.95]))

print("\n\nTraining XOR\n\n")
xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

xorn = NeuralNet(2,1,1)
xorn.train(xor_training_data)
print(xorn.test_with_expected(xor_training_data))

test_training_data =  [
    ([.9,.8,.6,.3,.1],[1]),
    ([.8,.8,.4,.6,.4],[1]),
    ([.7,.2,.4,.6,.3],[1]),
    ([.5,.5,.8,.4,.8],[0]),
    ([.3,.1,.6,.8,.8],[0]),
    ([.6,.3,.4,.3,.6],[0])
]
test_data = [
    ([1,1,1,.1,.1]),
    ([.5,.2,.1,.7,.7]),
    ([.8,.3,.3,.3,.8]),
    ([.8,.3,.3,.8,.3]),
    ([.9,.8,.8,.3,.6])
]
test = NeuralNet(5,8,1)
test.train(test_training_data)
print(f"Weights:{test.get_ho_weights()}")
print(test.test_with_expected(test_training_data))
for person in test_data:
    print(round((test.evaluate(person)[0])))