EQUATION
    # Define shapes for clarity (these would be parameters passed to the equation)
    state_shape = (state_size,)
    input_shape = (input_size,)
    output_shape = (output_size,)

    # Learnable parameters for discrete SSM
    LEARNABLE A_state = RandomNormal(shape=(state_size, state_size), mean=0.0, stddev=0.05)
    LEARNABLE B_input = RandomNormal(shape=(state_size, input_size), mean=0.0, stddev=0.05)
    LEARNABLE C_state = RandomNormal(shape=(output_size, state_size), mean=0.0, stddev=0.05)
    LEARNABLE D_direct = RandomNormal(shape=(output_size, input_size), mean=0.0, stddev=0.05)

    # Declare inputs: previous state and current input
    prev_state = INPUT(name='prev_state')
    curr_input = INPUT(name='curr_input')

    # Compute next state: x_k = Ā x_{k-1} + B̄ u_k
    next_state = Add(MatMul(A_state, prev_state), MatMul(B_input, curr_input))

    # Compute output: y_k = Ĉ x_k + D̄ u_k
    output = Add(MatMul(C_state, next_state), MatMul(D_direct, curr_input))

    # Concatenate state and output for returning both values
    state_output = Concatenate([next_state, output], axis=0)

    OUTPUT = state_output
END