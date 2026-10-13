EQUATION
    input_shape = (input_size,)
    # Fixed (non-trainable) parameters
    FIXED W1 = Zeros(shape=(input_size, hidden_size))
    FIXED b1 = Zeros(shape=(hidden_size))
    # Trainable parameters
    LEARNABLE W2 = RandomNormal(shape=(hidden_size, output_size), mean=0.0, stddev=0.05)
    LEARNABLE b2 = Zeros(shape=(output_size,))
    x = INPUT(shape=input_shape)
    y = Add(MatMul(x, W1), b1)
    z = Relu(y)
    output = Add(MatMul(z, W2), b2)
END