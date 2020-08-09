#Calculate b
def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff  
  return b_gradient

#Calculate m
def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff  
  return m_gradient

#Step gradient
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]

#Loop of gradient computation
def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)
        cost = (1/num_iterations)*sum([val**2 for val in (y-((m * x) + b))])
        print("b : {}, m : {}, cost : {}".format(b,m, cost))
    
    return [b, m]


import pandas as pd
import matplotlib.pyplot as plt

def main():

    df = pd.read_csv("heights.csv")
    X = df["height"]
    y = df["weight"]
    plt.plot(X, y, 'o')
    plt.show()
    b, m = gradient_descent(X, y, num_iterations=10000, learning_rate=0.0001)
    y_predictions = [m*x + b for x in X]
    plt.plot(X, y_predictions)
    plt.show()

if __name__ == "__main__":

    main()

