EQUATION
    input_shape = (input_size,)
    LEARNABLE W1 = RandomNormal(shape=(input_size, hidden_size), mean=0.0, stddev=0.05)
    LEARNABLE b1 = Zeros(shape=(hidden_size,))
    LEARNABLE W2 = RandomNormal(shape=(hidden_size, output_size), mean=0.0, stddev=0.05)
    LEARNABLE b2 = Zeros(shape=(output_size,))
    x = INPUT(shape=input_shape)
    y = Add(MatMul(x, W1), b1)
    z = Relu(y)
    output = Add(MatMul(z, W2), b2)
END