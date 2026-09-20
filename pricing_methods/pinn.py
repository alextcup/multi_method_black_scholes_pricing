import tensorflow as tf

from option import Option


def payoff(S, K, option_type):
    if option_type == "call":
        return tf.maximum(S - K, 0)
    elif option_type == "put":
        return tf.maximum(K - S, 0)


def build_model(option: Option, width, depth):
  K = option.K
  T = option.T

  model = tf.keras.Sequential(
      [tf.keras.layers.Input(shape=(2,)),
       tf.keras.layers.Rescaling(scale=[1.0 / K, 1.0 / T]),
      *[tf.keras.layers.Dense(width, activation='tanh') for _ in range(depth)],
       tf.keras.layers.Dense(1),
       tf.keras.layers.Rescaling(scale=K)
      ]
  )

  return model


def train(training_step, option, n_collocation, n_boundary, n_initial, Smin, Smax, epochs=1000, resample_rate=100, verbose=False):
  exercise = option.exercise
  for epoch in range(epochs):
    if epoch % resample_rate == 0:
        if exercise == "european":
            col_pts, bc_pts, ic_pts = sample_points_eur(option, n_collocation, n_boundary, n_initial, Smin, Smax)
        elif exercise == "american":
            col_pts, bc_pts, ic_pts = sample_points_am(option, n_collocation, n_boundary, n_initial, Smin, Smax)   
    PDE_loss, IC_loss, BC_loss, loss = training_step(col_pts, bc_pts, ic_pts)
    if verbose:
        if epoch % 100 == 0:
            print(f'Epoch {epoch} - PDE Loss {PDE_loss.numpy()} - IV Loss {IC_loss.numpy()} - BC Loss {BC_loss.numpy()} - Total Loss {loss.numpy()}')


def boundary_conditions_eur(option: Option):
    K = option.K
    T = option.T
    r = option.r
    div_yield = option.div_yield
    option_type = option.option_type

    uSmin, uSmax = None, None
    if option_type == "call":
        uSmin = lambda S, t: tf.zeros_like(t)
        uSmax = lambda S, t: S * tf.exp(-div_yield*(T-t)) - K * tf.exp(-r * (T - t))
    elif option_type == "put":
        uSmin = lambda S, t: K * tf.exp(-r * (T - t)) - S * tf.exp(-div_yield*(T-t))
        uSmax = lambda S, t: tf.zeros_like(t)

    return uSmin, uSmax


def sample_points_eur(option: Option, n_collocation, n_boundary, n_initial, Smin, Smax):
    K = option.K
    T = option.T
    option_type = option.option_type

    tmin = 0.0
    tmax = T

    uSmin, uSmax = boundary_conditions_eur(option)

    col_pts = tf.random.uniform((n_collocation, 2), minval=[Smin, tmin], maxval=[Smax, tmax])

    sample_init_pts = tf.random.uniform((n_initial, 2), minval=[Smin, tmax], maxval=[Smax, tmax])
    init_pts = payoff(sample_init_pts[:,0], K, option_type)

    sample_Smin_pts = tf.random.uniform((n_boundary, 2), minval=[Smin, tmin], maxval=[Smin, tmax])
    sample_Smax_pts = tf.random.uniform((n_boundary, 2), minval=[Smax, tmin], maxval=[Smax, tmax])

    Smin_pts = uSmin(sample_Smin_pts[:,0], sample_Smin_pts[:,1])
    Smax_pts = uSmax(sample_Smax_pts[:,0], sample_Smax_pts[:,1])

    ic_pts = tf.concat([sample_init_pts, tf.expand_dims(init_pts, axis=1)], axis=1)
    bc_pts = tf.concat([sample_Smin_pts, sample_Smax_pts, tf.expand_dims(Smin_pts, axis=1), tf.expand_dims(Smax_pts, axis=1)], axis=1)

    return col_pts, bc_pts, ic_pts


def make_training_step_eur(option: Option, model, optimizer, g_initial=1.0, g_boundary=1.0):
  r = option.r
  sigma = option.sigma
  div_yield = option.div_yield

  @tf.function
  def training_step(pts, bc_pts, init_pts):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(pts)
      V = model(pts)
      S = pts[:,0]

      grads = tape.gradient(V, pts)
      V_S, V_t = grads[:, 0], grads[:, 1]

      V_SS = tape.gradient(V_S, pts)[:, 0]

      eqn = V_t + 0.5 * sigma ** 2 * S ** 2 * V_SS + (r - div_yield) * S * V_S - r * tf.squeeze(V)

      PDE_loss = tf.reduce_mean(eqn**2)

      init_sample_pts = init_pts[:,0:2]
      init_true_pts = init_pts[:,2:3]
      IC_loss = tf.reduce_mean((init_true_pts - model(init_sample_pts))**2) # The loss of the initial condition

      bc_sample_Smin_pts = bc_pts[:, 0:2]
      bc_sample_Smax_pts = bc_pts[:, 2:4]
      bc_true_Smin_pts = bc_pts[:, 4:5]
      bc_true_Smax_pts = bc_pts[:, 5:6]

      BC_loss = tf.reduce_mean((model(bc_sample_Smin_pts) - bc_true_Smin_pts)**2 + (model(bc_sample_Smax_pts) - bc_true_Smax_pts)**2)

      loss = PDE_loss + g_initial * IC_loss + g_boundary * BC_loss # The loss function itself

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return PDE_loss, IC_loss, BC_loss, loss

  return training_step


def pinn_pricer_eur(option: Option, width=64, depth=4, epochs=1000, learning_rate=1e-3,
                    n_collocation=5000, n_boundary=1000, n_initial=1000,
                    g_initial=1.0, g_boundary=1.0, Smin=0.05, Smax=None, resample_rate=50,
                    verbose=False):
    if Smax is None:
        Smax = 3 * option.K

    S0 = option.S0

    model = build_model(option, width, depth)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    training_step = make_training_step_eur(option, model, optimizer, g_initial, g_boundary)
    train(training_step, option, n_collocation, n_boundary, n_initial, Smin, Smax,
          epochs=epochs, resample_rate=resample_rate, verbose=verbose)

    return float(model(tf.constant([[S0, 0.0]], dtype=tf.float32))[0, 0])


def boundary_conditions_am(option: Option):
    K = option.K
    option_type = option.option_type

    uSmin, uSmax = None, None
    if option_type == "call":
        uSmin = lambda S, t: tf.zeros_like(t)
        uSmax = lambda S, t: S - K
    elif option_type == "put":
        uSmin = lambda S, t: K - S
        uSmax = lambda S, t: tf.zeros_like(t)

    return uSmin, uSmax


def sample_points_am(option: Option, n_collocation, n_boundary, n_initial, Smin, Smax):
    K = option.K
    T = option.T
    option_type = option.option_type

    tmin = 0.0
    tmax = T

    uSmin, uSmax = boundary_conditions_am(option)

    col_pts = tf.random.uniform((n_collocation, 2), minval=[Smin, tmin], maxval=[Smax, tmax])

    sample_init_pts = tf.random.uniform((n_initial, 2), minval=[Smin, tmax], maxval=[Smax, tmax])
    init_pts = payoff(sample_init_pts[:,0], K, option_type)

    sample_Smin_pts = tf.random.uniform((n_boundary, 2), minval=[Smin, tmin], maxval=[Smin, tmax])
    sample_Smax_pts = tf.random.uniform((n_boundary, 2), minval=[Smax, tmin], maxval=[Smax, tmax])

    Smin_pts = uSmin(sample_Smin_pts[:,0], sample_Smin_pts[:,1])
    Smax_pts = uSmax(sample_Smax_pts[:,0], sample_Smax_pts[:,1])

    ic_pts = tf.concat([sample_init_pts, tf.expand_dims(init_pts, axis=1)], axis=1)
    bc_pts = tf.concat([sample_Smin_pts, sample_Smax_pts, tf.expand_dims(Smin_pts, axis=1), tf.expand_dims(Smax_pts, axis=1)], axis=1)

    return col_pts, bc_pts, ic_pts


def make_training_step_am(option: Option, model, optimizer, g_initial=1.0, g_boundary=1.0):
  K = option.K
  r = option.r
  sigma = option.sigma
  div_yield = option.div_yield
  option_type = option.option_type

  @tf.function
  def training_step(pts, bc_pts, init_pts):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(pts)
      V = model(pts)
      S = pts[:,0]

      grads = tape.gradient(V, pts)
      V_S, V_t = grads[:, 0], grads[:, 1]

      V_SS = tape.gradient(V_S, pts)[:, 0]

      eqn = V_t + 0.5 * sigma ** 2 * S ** 2 * V_SS + (r - div_yield) * S * V_S - r * tf.squeeze(V)
      res = tf.minimum(-eqn, tf.squeeze(V) - payoff(S, K, option_type))

      PDE_loss = tf.reduce_mean(res**2)

      init_sample_pts = init_pts[:,0:2]
      init_true_pts = init_pts[:,2:3]
      IC_loss = tf.reduce_mean((init_true_pts - model(init_sample_pts))**2) # The loss of the initial condition

      bc_sample_Smin_pts = bc_pts[:, 0:2]
      bc_sample_Smax_pts = bc_pts[:, 2:4]
      bc_true_Smin_pts = bc_pts[:, 4:5]
      bc_true_Smax_pts = bc_pts[:, 5:6]

      BC_loss = tf.reduce_mean((model(bc_sample_Smin_pts) - bc_true_Smin_pts)**2 + (model(bc_sample_Smax_pts) - bc_true_Smax_pts)**2)

      loss = PDE_loss + g_initial * IC_loss + g_boundary * BC_loss # The loss function itself

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return PDE_loss, IC_loss, BC_loss, loss

  return training_step


def pinn_pricer_am(option: Option, width=64, depth=4, epochs=1000, learning_rate=1e-3,
                   n_collocation=5000, n_boundary=1000, n_initial=1000,
                   g_initial=1.0, g_boundary=1.0, Smin=0.05, Smax=None, resample_rate=50,
                   verbose=False):
    if Smax is None:
        Smax = 3 * option.K

    S0 = option.S0
    K = option.K
    option_type = option.option_type

    model = build_model(option, width, depth)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    training_step = make_training_step_am(option, model, optimizer, g_initial, g_boundary)
    train(training_step, option, n_collocation, n_boundary, n_initial, Smin, Smax,
          epochs=epochs, resample_rate=resample_rate, verbose=verbose)

    price = float(model(tf.constant([[S0, 0.0]], dtype=tf.float32))[0, 0])
    intrinsic = float(payoff(S0, K, option_type))

    return max(price, intrinsic)


def pinn_pricer(option: Option, **kwargs):
    if option.exercise == "european":
        return pinn_pricer_eur(option, **kwargs)
    elif option.exercise == "american":
        return pinn_pricer_am(option, **kwargs)
