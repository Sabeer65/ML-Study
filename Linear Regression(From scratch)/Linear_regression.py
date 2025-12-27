import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('studytime.csv')

def loss_function(m,b,points): # not being used as gradient descent has decent error calculation
    total_errors = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
    
        total_errors += (y - (m * x +b)) **2
        total_errors = total_errors/float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += (2 / n) * x * ((m_now * x + b_now) - y)
        b_gradient += (2 / n) * ((m_now * x + b_now) - y)
    
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m,b 

m = 0
b = 0
L = 0.0001 
epochs = 2000

# Calculation for the gradient descent 

for i in range(epochs):
    if i % 50 == 0:
        print(f'Epoch: {i}')
    m, b = gradient_descent(m,b,data,L)

print(m,b)


x_line = [data.studytime.min(), data.studytime.max()]
y_line = [m * x_line[0] + b, m * x_line[1] + b]

plt.scatter(data.studytime, data.score, color="black")
plt.plot(x_line, y_line, color="red", linewidth=2)
plt.show()