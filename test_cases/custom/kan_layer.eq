EQUATION
    input_shape = (input_size,)
    output_shape = (output_size,)

    # Learnable coefficients for quadratic basis functions:
    # For each output dimension i and input dimension k:
    #   phi_{ik}(x) = c2_{ik} * x^2 + c1_{ik} * x + c0_{ik}
    LEARNABLE coeffs_c2 = RandomNormal(shape=(output_size, input_size), mean=0.0, stddev=0.1)
    LEARNABLE coeffs_c1 = RandomNormal(shape=(output_size, input_size), mean=0.0, stddev=0.1)
    LEARNABLE coeffs_c0 = RandomNormal(shape=(output_size, input_size), mean=0.0, stddev=0.1)

    x = INPUT(shape=input_shape)
    # Compute squared inputs (requires element-wise multiplication)
    x_squared = Multiply(x, x)

    # Compute KAN layer output:
    # output[i] = sum_k [ coeffs_c2[i,k] * x[k]^2 + coeffs_c1[i,k] * x[k] + coeffs_c0[i,k] ]
    term1 = MatMul(coeffs_c2, x_squared)  # Weighted x^2 terms
    term2 = MatMul(coeffs_c1, x)          # Weighted x terms
    term3 = MatMul(coeffs_c0, Oneshot(shape=input_size, value=1.0))  # Bias terms
    output = Add(Add(term1, term2), term3)
END