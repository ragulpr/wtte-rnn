
def weibull_logLik_continuous(a_, b_, y_, u_,name=None):
    ya = tf.div(y_+1e-35,a_) # Small optimization y/a
    return(
        tf.mul(u_,
               tf.log(b_)+tf.mul(b_,tf.log(ya))
              )- 
        tf.pow(ya,b_)
    )

def weibull_logLik_discrete(a_, b_, y_, u_, name=None):
    with tf.name_scope(name):
        hazard0 = tf.pow(tf.div(y_+1e-35,a_),b_) # 1e-9 safe, 1e-37 is min float
        hazard1 = tf.pow(tf.div(y_+1,a_),b_)
    return(tf.mul(u_,tf.log(tf.exp(hazard1-hazard0)-1.0))-hazard1)       

def weibull_beta_penalty(b_,location = 10.0, growth=20.0, name=None):
    # Regularization to keep beta below location
    # becomes (positive) large when beta approaches location
    with tf.name_scope(name):
        scale = growth/location
        penalty_ = tf.exp(scale*(b_-location))
    return(penalty_)       
