


[Open source repository](https://github.com/VeritNet/AI-Learning) [Website](veritnet.com)
<p xmlns:cc=" http://creativecommons.org/ns# " xmlns:dct=" http://purl.org/dc/terms/ "><span property="dct:title">BINKLINGS AI learning  © 2023-2024</span> is licensed under <a href=" http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser -v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block; ">CC BY-NC-SA 4.0</a>. To view a copy of this license, visit  http://creativecommons.org/licenses/by-nc-sa/4.0 </p>
[Website](https://www.binklings.com) [Youtube Channel](https://www.youtube.com/@BINKLINGS)
# Preface

## <h3 style="color: red">Guide to Reading This Book (Required)</h3>

### Common Symbols and Fonts

Here's a guide to common symbols and fonts used in this book:

**Bold:** **Important terms (Full English name / English abbreviation)**
*Italics:* *Mathematical formulas a+b=c*
Code blocks:

```cpp
// This is some C++ code
```

Underlined: ++Important content++ (It is recommended that you browse through all the underlined and boldfaced words in a chapter after reading it completely. This will help you understand the content from a global and holistic perspective.)

### About Code and Practice

The practice in this book is divided into two parts: code practice and GeoGebra mathematical animation practice. Both are very important.

When I present a link to my GeoGebra work, I highly recommend you open it and try to interact with it (e.g., manually change a value and see what happens to the image). It will help you better understand some of the theories.

The code I use is primarily JavaScript and C++. I use them for the following reasons:

- JavaScript is concise and closer to natural mathematical language when performing various calculations. As long as your device has a browser, whether it's a mobile device or a computer, it can run JavaScript code.
- Some of the syntax in C++ is common with JavaScript. C++ runs tens or even hundreds of times faster than other languages.

I don't recommend using Python. Although its ecosystem is more complete and it might be easier to learn, in the long run, no matter how many calculation acceleration libraries you call, it's not as fast as C++.

Finally, I don't recommend calling too many math libraries either, even though they make things much easier. This wastes huge hardware resources, and our principle is to make the most of the hardware resources we already have.

**Note:**

If your C++ compiler supports versions below C++11 or doesn't support list initialization, please change all instances in the C++ code from:

```cpp
std::vector<double> xxxxxx = {1.0, 2.0, 3.0};
```

to this format:

```cpp
std::vector<double> xxxxxx;
xxxxxx.push_back(1.0);
xxxxxx.push_back(2.0);
xxxxxx.push_back(3.0);
```

### Basic Knowledge

Whether you are an elementary school student or a graduate student, this book is suitable for you. This book introduces some advanced mathematical knowledge, but you need to be prepared with the following:

- Basic knowledge of functions (required content in junior high school textbooks)
- Basic knowledge of derivatives (3Blue1Brown's explanation of derivatives is excellent, I suggest you watch his series of videos: YouTube: https://www.3blue1brown.com/topics/calculus.
- At least one programming language (recommended: C++, preferably with some knowledge of JavaScript). 

# Chapter 1 Introduction to Neural Networks

## Section 1 From Perceptron to Neuron

### 1.1 Perceptron

Let's say we have this problem: We all know that whether people engage in outdoor activities is closely related to the current temperature. If we collect some data, i.e., whether people choose to be outdoors at a given temperature x, and plot them on a coordinate system, it might look like this:
![ ](./images/1694611799051.jpg)

Where 0 on the vertical axis represents inactivity and 1 represents activity. Now, if we want to predict the probability of a person going outdoors based on a given temperature, how do we do that? We would, of course, first draw a dividing point for these data points, and to the left of it, predict no outdoor activity, and to the right of it, the opposite.
If we need to get the computer program to find this point, how do we do that? If you've studied statistics, you probably know all sorts of regression algorithms, but today we're going to think about it differently.
Below is a **Perceptron**:

![ ](./images/1693394603517.png)

It was first proposed by computer scientist Rosenblatt in 1958.  Before we talk about the principle, let's write its algorithm from God's perspective: 
*output=f(∑j wjxj+b)* 
If written in a popular way, it is exactly equal to this formula:
*output=f(w1×x1+w2×x2+...+wn×xn+b)*

++w and b are both attributes, or parameters, of the perceptron itself, while x is the input value of the perceptron.  The f function will determine whether the data is greater than 0 or less than 0, returning 1 if it is greater than 0 and 0 if it is less than 0++.  Yes, you can temporarily think of the perceptron as a complex function, we input some values into it (x1,x2...xn), it processes them and returns 0 or 1.  This processing is done by formula 1, ++multiplying all input values x by their corresponding **weights**, adding up these products and finally adding the **bias**.++ (w and b are their abbreviations, don't worry about it).  Note that in each perceptron, ++there is a corresponding weight coefficient w for each input value x, but only one bias coefficient b for a single perceptron++.  Isn't this just the linear function we learned in junior high school plus an f function?  If we look at the original "Outdoor Activity Probability vs. Temperature Relationship Graph," it is not difficult to see that a single perceptron can be thought of as a function of output with respect to x, with w and b being the coefficients.  In this way, as long as the appropriate w and b are found, a function line (or surface, or hypersurface) can be drawn, but this line (or surface, or hypersurface) can only perform binary classification and cannot accomplish our task.

According to the example just given, the horizontal axis represents temperature, the blue dots represent data, and the vertical axis represents whether or not to go out for activities. The orange function line separates "suitable for outdoor activities" from "not suitable for outdoor activities", and the position of the "dividing point" between the two varies with w and b (because there is only one dimension, so it varies left and right). Below is an animated demonstration of the classification work of this perceptron (you can drag the slider in the upper left corner with your mouse or finger to change the values of w and b): [https://www.geogebra.org/m/gwjweugy](https://www.geogebra.org/m/gwjweugy).  It is imperative that you experience all such demonstrations in this book firsthand in order to understand them well!
This is the screen you will be able to experience after you open it:
![ ](./images/1694702138730.png)

You should also have the ability to think about extending it to higher dimensions:
![ ](./images/1694702023153.png)



At this point, we should have some understanding of the original form of a single perceptron.  In summary, it first draws a straight line (or plane, or hyperplane) using an equation, then plugs the new data into the perceptron function, the resulting value is judged by the function f for positive or negative, and finally outputs the output value 0 or 1, representing which category the new data falls into.

We also found a number of problems: ++The perceptron can only perform binary classification, and it is **linear**, meaning that the line (or plane, etc.) drawn by that function is a straight line or plane, not a curve or surface.  This is also easy to prove, we will verify this again later, but now you can also use pen and paper to verify it, like this (I omitted the bias b first):
![ ](./images/1693474767897.jpg)

We ++take the output of a single perceptron as the input value of the next perceptron++, we get the structure above.  Plotting the function image corresponding to the last part of the formula, you will always get only a linear image, no matter how many perceptrons are connected.
This is because, no matter how many x's there are, they are only constantly multiplied by a constant w and then added together, so their ++exponent will always be 1++, i.e., the first power, and we know that a linear function is always linear.
In fact, this problem once led to the denial of the perceptron.  Because it ++cannot solve the **XOR problem**++


![ ](./images/1693457556412.png)
As shown in the figure, can we find a straight line to perfectly separate the red and blue data points?  Obviously not.
At this point, the protagonist of our day is coming.  Let's first talk about the **Neuron** that makes up the entire neural network.

### 1.2 Neuron

I have drawn a diagram showing the structure of a single neuron in a neural network:


![ ](./images/1693472595158.png)
It's still a lot like a perceptron, take a closer look, what are the specific changes from a perceptron?  You will most likely notice that the function f, which was originally used to determine positive or negative, is gone. This means that the new neuron will not only output 0 or 1.  In fact, it can output any value, such as 0.11, 0.94 and 0.56 and so on.  Instead of the function f, there is an Activate operation, which is the right half of the circle in the image.
The expression for the new neuron: *output=f(∑j wjxj+b)*, where *f(x)=Activate(x)*.  The function symbol f is still used here, and some books use the symbol σ(x), but remember that this function is no longer used to determine positive or negative, but rather an activation operation.
As an aside, in some standard textbooks, the bias b is written on the image as w0 multiplied by a part of the input of the entire neuron; more often, we are used to writing the bias coefficient b directly next to the neuron, and for the sake of looking comfortable, we sometimes write all the weight coefficients w inside the neuron (or hidden), rather than on the input arrows as in the image.
Back to the topic at hand, this ++activation operation is actually also a function called the **Activation Function**, and the output value of the neuron calculated by the activation function is called the **activation**++.  What does this function do?  We just said that the perceptron can't solve the XOR problem because it is linear.  So can the activation function solve this problem?  Of course it can! In order to make that one dividing or fitting line (or surface, or hypersurface) complex and diverse to fit or classify complex data, we have to make sure that the dividing or fitting line of the neuron is a curve (I probably won't need to emphasize "line (or surface, or hypersurface)" again, and you should be able to think of this yourself when you see me say dividing line later), in technical terms, ++the activation function must be non-linear++.
Let's take a look at a few common activation functions.

This is the **ReLU activation function (The Rectified Linear Unit)**:
```
f(x) = max(0,x)
```
This is its function image:

![ ](./images/1693475859793.jpg)
If the input value is less than 0, it returns 0, if the input value is greater than 0, it returns the input value itself. It looks simple, and it's the most popular activation function! (But because of its characteristic of returning 0 when the input is negative, it can lead to the permanent death of some neurons when the learning rate is too high, which sounds terrible, what does it mean? We will learn later that there are activation functions such as Leaky ReLU, PReLU, ELU, SeLU and GeLU to solve this problem, but please don't worry about this in advance).

If you have learned javascript, you can write ReLU as the following code (js language is very simple, generally you can understand it even if you have not learned it):
```javascript
function relu(z) {
    return Math.max(0, z);
}
```

Here is the c++ code expression of the ReLU function:
```cpp
float relu(float x) {
    return (x > 0) ? x : 0;
}
```

This is the **Sigmoid activation function (S-function / S-shaped growth curve)**:
(Since some markdown browsers cannot properly accommodate mathematical expression formats, some mathematical formulas are presented using images to ensure that you can see them clearly)
![ ](./images/1693484723456.png)
Don't be intimidated, take a closer look, it's actually quite simple.  The "e" symbol in the function is just a mathematical constant, similar to the pi (π) we learned in elementary school, and has a fixed value.
*e≈2.718281828459*
If you can't understand why there is an e coming here, don't worry about it yet, we'll get to it later.  Plot this function, it looks like this:
![ ](./images/1698826249512.png)


Interestingly, this function scales (maps) any input value to a continuous output between 0 and 1, and only when the input value changes around 0 will the output change significantly, otherwise, it will either be close to 0 or close to 1.
This function is the first one taught in most tutorials and is also more popular, but the biggest problem is: it can cause the gradient to disappear in deep neural networks. (You don't have to understand this now, I'm just giving you a heads up)
++Neurons with Sigmoid as the activation function are called **sigmoid neurons**++

This is its js code expression:
```javascript
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
```

This is the c++ version:
```cpp
#include <cmath>

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}
```

This is the **Tanh activation function (hyperbolic tangent function)**:
![ ](./images/1693559043608.png)
Actually what you see is the expansion of the final expression `tanh(x)=sinh(x)/cosh(x)`, we won't look at this complex function analytical expression, let's go straight to the graph:
![ ](./images/1698826335568.png)


Its image is simply too much like Sigmoid, except that it maps, or scales, any number between -1 and 1, while Sigmoid is between 0 and 1.  Its distribution is more symmetrical.  It can be seen that it will also lead to the problem of vanishing gradients, which is a later story.
Its js code expression is:
```javascript
function tanh(x) {
  // Calculate tanh using the exponential function
  var e1 = Math.exp(x);
  var e2 = Math.exp(-x);
  return (e1 - e2) / (e1 + e2);
}
```

c++'s cmath has a ready-made tanh function: (cmath has almost zero impact on program performance, or 100% no impact under normal circumstances)
```cpp
#include <cmath>

double tanh(double x) {
  // Calculate tanh using the standard library function
  return std::tanh(x);
}
```

I recommend that you remember the two most common activation functions: ReLU and Sigmoid.
There are many other activation functions, such as ELU and Softmax, but they are not the focus now, so you don't have to worry about them.

This is a neuron in a neural network.  In summary, a neuron is an optimized version of a perceptron, it can output any value with decimals, and through the activation function, it achieves the nonlinearity of the dividing line, allowing it to begin to adapt to more complex data and preliminarily solve simple XOR problems.

Here is the js code expression of a single neuron (which has been carefully broken down for learning):
```javascript
function Neuron(w, x, b) {
    function relu(z) {
        return Math.max(0, z);
    }
    function sigma(w, x, b) {
        return relu(w.reduce((acc, curr, i) => acc + curr * x[i], 0) + b);
    }
    return sigma(w, x, b);
}

//Example usage
console.log(Neuron([0.1,0.6],[0.4,0.3],0.5))//The output value is 0.72
//Equivalent to x1 being 0.4, x2 being 0.3, w1 being 0.1, w2 being 0.6, and bias b being 0.5
//The calculation process is equivalent to 0.1 × 0.4+0.6 × 0.3+0.5=0.72
```

c++ version:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

double relu(double z) {
    return std::max(0.0, z);
}

double sigma(std::vector<double> w, std::vector<double> x, double b) {
    double sum = 0;
    for (int i = 0; i < w.size(); ++i) {
        sum += w[i] * x[i];
    }
    return relu(sum + b);
}

double Neuron(std::vector<double> w, std::vector<double> x, double b) {
    return sigma(w, x, b);
}

int main() {
    std::vector<double> w = {1.0, 2.0, 3.0};
    std::vector<double> x = {4.0, 5.0, 6.0};
    double b = 7.0;
    double result = Neuron(w, x, b);
    std::cout << result << std::endl;//The output value is 39
    return 0;
}
//The principle is the same as the JS version
```

Although c++ is difficult to write, I still recommend that you give it a try, because its performance and speed will far exceed other languages in the various deep learning neural networks you will learn in the future!
In the next section, we'll see how neurons are constructed into a neural network.


## Section 2 Neurons Make Up a Neural Network

### 2.1 The Architecture of a Neural Network

![ ](./images/1693572500892.png)
This is the most basic neural network, the ++**Feedforward Neural Network (FNN)**, pictured is a **Multilayer Perceptron (MLP)**, which is one of the most common types of feedforward neural networks, which in turn are the most common type of neural network.  MLPs are sometimes called **Fully Connected Neural Networks (FCNN)**++.
In this MLP neural network, ++the output value output of each neuron is used as the input value x of the next neuron to participate in the calculation of the next neuron, and so on, the neurons are fully connected, the output of one neuron may be used as input by multiple neurons, and one neuron may also input the output values of multiple neurons.  Numerical values are constantly processed by each neuron and propagated forward, from the **input layer** of the neural network (the leftmost **Input Layer**) to the **output layer** of the neural network (the rightmost **Output Layer**), and the neurons in between form the **Hidden Layers**++.
Please note! ++The multilayer perceptron MLP is not made up of perceptrons, but of the new type of neurons we talked about later++.  The term "multilayer perceptron" has become a colloquial term, although technically we know that it is actually composed of multiple neurons.  The use of this terminology helps to distinguish between single-layer perceptrons and multilayer structures, and highlights the ability of multilayer networks to handle complex problems.

We know that even with non-linear activation functions, the capabilities of a single neuron are limited, for example, if it is only fitting or classifying five or six simple pieces of data, even a single neuron can make this perfect dividing line, but we know that in many cases, the data is incredibly complex, and the number of input values x is large (x1,x2,...x9999...), which also increases the difficulty of division.  Neural networks, or MLPs, by increasing the number of neurons and fully connecting them, have achieved the ability to make dividing or fitting lines of almost any shape, or rather, to fit almost any dividing or fitting function.

++For example, we talked about a single neuron *output=f(∑j wjxj+b)* earlier, let's set the entire calculation of a single neuron as a function function Neural_X_X(\[x\]), where lowercase x is an array, representing all input values x1, x2, xn...; the first uppercase X in the function name represents whether it is a hidden layer or an output layer, H is written for the hidden layer, O is written for the output layer, and the second represents which neuron it is in that layer, for example, the function name of the first neuron in the hidden layer is *Neural_H_1*. Let's assume that all the parameters (the various weight coefficients w and a bias coefficient b) are already contained in their respective neuron functions, and let's assume that they have been adjusted to the most appropriate values (which can only be assumed, after all, we haven't learned how to adjust them yet).  Now look at the diagram above again, this Input layer does not have actual neuron calculations, they are all input data x, which means that this neural network has three input values: x1, x2, x3.  Now look at one of the neurons in the Hidden layer, it is connected to every piece of input data, which is x1, x2, x3, so the expression for its output value is: *Hidden_Neural_1_Output=Neural(x1,x2,x3)*, here I use Hidden_Neural_1_Output to represent the output value of the first neuron in the hidden layer, we will abbreviate H1 to represent the output value of the first neuron in the hidden layer for convenience, H2 is the second, and similarly H3 and H4 (we can see from the figure that there are only 4 neurons in the hidden layer of this neural network), abbreviate O1 as the output value of the first neuron in the output layer, and O2 as the second.  Then we can write *H1=Neural_H_1(x1,x2,x3)*, *H2=Neural_H_2(x1,x2,x3)*, H3 and H4 are analogous.  If we look at the output layer, the input value of each neuron in it is actually the output value of each neuron in the previous hidden layer, so we have *O1=Neural_O_1(H1,H2,H3,H4)*, *O2=Neural_O_2(H1,H2,H3,H4)*.  In this way, we can think of the entire neural network as a super complex function! We modify and adjust the parameters w and b of each neuron to modify and adjust the shape of this function to make it the closest thing to a perfect dividing or fitting line. Having said that, do you have some of your own insights into the multilayer perceptron MLP and feedforward neural network (forward propagation) we just mentioned?  Do you understand what it means for data to flow forward (forward propagation/feedforward)?++

Now you should try this demo animation again, in this new demo animation, we set up 1 input neuron (the input neuron does not process the data in any way), 2 hidden layer neurons and 1 output layer neuron, the output value corresponds to the y-axis in the image, and the input value corresponds to the x-axis.  All neurons use the sigmoid activation function.  Here is the link to the demo animation: [ https://www.geogebra.org/m/sqbm5quf](https://www.geogebra.org/m/sqbm5quf) (you can use your mouse or finger to drag the sliders in the upper left corner to change the values of the parameters w and b of each neuron).

You'll notice that as you change all the parameters w and b, the shape, position, etc. of the function will change.  You can design all sorts of shapes just by adjusting these parameters.
![ ](./images/1694773843206.jpg)

Try replacing the sigmoid function with the other activation functions we mentioned earlier, such as ReLU, and see what happens? (You can change the original expression after *sigmoid =* to *If(x > 0, x, If(x < 0, 0))* which is one way to express the ReLU activation function). Is there a subtle relationship between the shape of the prediction function and the activation function?
![ ](./images/1694773983783.png)

However, I would like to say that this is still too limited, MLP has only 1 input and output layer, but can have multiple hidden layers, in the feedforward neural network, they are called multilayer feedforward neural networks.  Please note! A single-layer feedforward neural network has only one output layer neuron, so it does not constitute an MLP! We will not learn it because it simply does not meet our needs.  It is important not to confuse the concepts of "single layer" and "single hidden layer". Again, MLP is a type of multilayer feedforward neural network, must have an input layer and an output layer, and has at least one or more hidden layers.

In the example above, we only have a 3-layer neural network and 2 hidden layer neurons, only 7 adjustable parameters, and the data dimension is only 1, so it is not possible to achieve fitting any function. But imagine a deep neural network with dozens or even hundreds of layers, a model with hundreds of millions or even trillions of parameters, what a spectacular prediction function it would draw!

But without the activation function, everything becomes bland, try removing the non-linear part of the activation function, for example, replacing sigmoid(x) = ... with x, i.e. returning x itself, then the prediction function will always be like this:
![ ](./images/1694949627142.jpg)

Just like that, we have once again validated the biggest advantage of neurons over perceptrons: solving XOR and non-binary classification problems.

While you are still familiar with the new knowledge, let's go back to the problem at the beginning of this book and see if we have any ideas to solve it now?
![ ](./images/1693635604732.png)
Now, the data looks a little crazy, obviously we would need to use a curve if we wanted to fit this data for prediction. By the way, ++this curve is called the **Prediction function**++.  This is familiar to us, we only need one with 1 input neuron, several hidden layer neurons plus one output layer neuron.  Among them, the input neuron inputs the value of "temperature", and the output neuron will output the "probability of outdoor activity".  But here I drew the y-axis as the probability of each data point instead of the exact 0 or 1, I hope this will make it easier for you to imagine the shape of this prediction function, but remember that in reality the data is usually not like this.

Here I have to say again that the prediction function of an ordinary neural network is an ++explicit function++ of the output value with respect to the input value, that is, there is a unique dependent variable corresponding to the independent variable, and ++it is not an implicit function or equation++.  If you see a prediction function that is closed or does not look like an explicit function, such as tensorflow playground, please do not misunderstand, the curve you see is not the original prediction function, but the projection (or intersection curve) of the three-dimensional prediction function surface on the threshold plane. Of course, if you prefer, you can also make the three-dimensional height equal to 0 to get an equation, which is indeed a closed dividing curve, but keep in mind that this is just making the height equal to 0, this curve is not the prediction function, the prediction function here is still three-dimensional.
![ ](./images/1694845138835.png)
For example, the blue and orange here represent the height of the corresponding position of the prediction surface, not that it itself is the prediction function.  The green surface in the figure below is the real prediction function, and the plane you see is the height of the corresponding position of the green curved surface on the blue plane, that is, the green curved surface above the blue plane in the figure below corresponds to the orange part in the figure above, and the one below the blue plane is the opposite.
![ ](./images/1694845300706.png)
Now, we have roughly learned the architecture of neural networks. Almost forgot to introduce, Tensorflow is a neural network library developed by Google, and playground is a great demo tool for learning neural networks. http://playground.tensorflow.org Here is the link to the playground, and as we learn later on, you will gradually understand the principles of this demo.

You must be eager to try to build your own neural network, but we haven't learned how to adjust the correct parameters w and b to make this prediction function. But I've already pre-trained a model, you can play with it first and get a feel for what we just learned.
Let's write out the neural network code first, here because the neural network is so small, we don't need to use loops to calculate it.
This is the js version:
```javascript
//The first neuron in the hidden layer
na1w1 = -0.16132499999994152
na1w2 = 0.7
na1b = 0.07210999999999113
//The second neuron in the hidden layer
na2w1 = -0.2150150000000196
na2w2 = 0.8
na2b = 1.4037740000000716
//Output layer neurons
outw1 = 0.3427469999998477
outw2 = -0.7286400000000355
outb = 0.6238199999996561

pv = predict(18)//The probability of people going out for activities when the temperature is n ℃ (estimated temperature range: 0-30 ℃)
console.log("Predict activity probability:"+Number(pv).toFixed(2))//The output value is 0.76

//Neuronal computation
function Neuron(w, x, b) {
    //Tanh activation function
    function tanh(z) {
        var e1 = Math.exp(z);
        var e2 = Math.exp(-z);
        return (e1 - e2) / (e1 + e2);
    }
    //Sum of continuous multiplication
    function sigma(w, x, b) {
        return tanh(w.reduce((acc, curr, i) => acc + curr * x[i], 0) + b);
    }
    
    return sigma(w, x, b);
}
//Prediction function, i.e. forward propagation
function predict(content){
    in1 = content
    
    na1 = Neuron([na1w1,na1w2],[in1],na1b)
    na2 = Neuron([na2w1,na2w2],[in1],na2b)
    out = Neuron([outw1,outw2],[na1,na2],outb)
    
    return String(out);
}
```

This is the c++ version:
```cpp
#include <iostream>
#include <cmath>

using namespace std;

//The first neuron in the hidden layer
double na1w1 = -0.16132499999994152;
double na1w2 = 0.7;
double na1b = 0.07210999999999113;
//The second neuron in the hidden layer
double na2w1 = -0.2150150000000196;
double na2w2 = 0.8;
double na2b = 1.4037740000000716;
//Output layer neurons
double outw1 = 0.3427469999998477;
double outw2 = -0.7286400000000355;
double outb = 0.6238199999996561;

//Tanh activation function
double tanh(double z) {
    double e1 = exp(z);
    double e2 = exp(-z);
    return (e1 - e2) / (e1 + e2);
}

//Sum of continuous multiplication
double sigma(double w[], double x[], double b) {
    double sum = 0.0;
    for (int i = 0; i < 2; i++) {
        sum += w[i] * x[i];
    }
    return tanh(sum + b);
}

//Neuronal computation
double Neuron(double w[], double x[], double b) {
    return sigma(w, x, b);
}

//Prediction function, i.e. forward propagation
double predict(double content) {
    double in1 = content;
    double na1 = Neuron(new double[2]{na1w1, na1w2}, &in1, na1b);
    double na2 = Neuron(new double[2]{na2w1, na2w2}, &in1, na2b);
    double out = Neuron(new double[2]{outw1, outw2}, new double[2]{na1, na2}, outb);
    return out;
}

int main() {
    double pv = predict(18);//The probability of people going out for activities when the temperature is n ℃ (estimated temperature range: 0-30 ℃)
    cout << "预测活动概率：" << pv << endl;//The output value is 0.76229
    
    return 0;
}
```

I only used 10 training data directly between 0 and 30 degrees to train this network, so you only get a more reliable output value when you input between 0 and 30.  And the neural network also has only 4 neurons.
I suggest you read the code (if you know js or c++) and see if each line of code corresponds to what we learned earlier.

### 2.2 Various Neural Networks

Before we learn how to train a neural network, let's go over a few confusing concepts that can be confusing to search for online, and many of the explanations in the text or videos are problematic.  I was also very disturbed in the process of learning, so I went back and reviewed a lot of literature and summarized the patterns in the terminology myself, drawing such a relational mind map:
![ ](./images/1693654283495.png)
CNN and RNN are actually BP neural networks, but it might be more convenient to write them this way in the diagram.  We haven't learned some of these concepts yet, but we should get to them later. Now we have learned the concepts of perceptron, neuron, and multilayer perceptron MLP.  Next we will learn how to train a neural network.


## Section 3 Training Our Neural Network

### 3.1 Loss and Cost

++The training process of a neural network is actually the continuous optimization of the prediction function.++
![ ](./images/1694844636677.png)
We need to prepare some data first, and then let the neural network find an optimal prediction function to fit or divide the data.  When we randomly initialize the parameters (weights and biases) of the neural network, it will randomly output a result when we input a piece of data.  For example, I initialized a neural network with 2 input neurons, n hidden neurons, and 1 output neuron, I want to input the outdoor temperature and humidity to let it tell me if it is suitable for outdoor exercise today (0 means not suitable, 1 means suitable), when I input 35 (here I am using Celsius) and 0.9 (humidity percentage), the neural network justified the output of 0.96, while I think this value should only be 0.2.  At this time, there is an **error**, or **loss**, between the ++output value and the true value of the prediction function++, both of which are a specific numerical value, usually the absolute value of the difference between the true value and the predicted value, i.e. *err=|y-y^|*, where y is the true value of a single piece of data, y^ is the predicted value (output value) of the neural network prediction function for that data, and the ^ symbol is read as hat.  If the neural network has multiple output neurons, i.e., multiple output values, you can ++take the average or use a **loss function** for each piece of data and sum them up++. However, both loss and loss function are for a single piece of data.  Each training of a neural network usually has many pieces of training data, in this case, in order to calculate the sum of errors of all the training data, we define **cost** and **cost function**.  You just need to remember: ++loss is for a single sample, cost is for all samples in a single training++.

Since we want to optimize and improve our model parameters, we need to know what is wrong with the model parameters and how much is wrong.
Now we need to learn some of the most basic cost functions:

1. **Mean square error (MSE)**
This cost function ++corresponds to the loss function should be the **Least-square method (LS)**++, "least" means ++finding the minimum value of the function++, "square" means that ++the error should be squared++. ![ ](./images/1694849005422.png) As shown in the figure, the input value x of data A is 1.75, the true value y is 0.87, and the predicted value is 3.06, then the error is *(0.87-3.06)²*, which is the square of l in the figure, 0.87². The least-squares method here can be simply written as *err=(y-y^)²*.  Now, after reading this sentence, pause and think for yourself, what are the advantages of the least-squares method?
The least-squares method, on the one hand, squares all the values of y-y^ ++into positive numbers++ to ensure that the errors are not cancelled out by each other when there are multiple data points in the future, and on the other hand, ++expands large errors and reduces small errors++, such as 0.1 squared becomes 0.01, 10 squared becomes 100, thus making key errors stand out, and most importantly, ++the absolute value is not differentiable over the entire range++ (the top of the absolute value image is pointed and not differentiable, while the top of the quadratic function is rounded and differentiable), we will understand the importance of this when we learn about gradient descent later.
Now we have a lot of training data, as shown in the following figure:
![ ](./images/1694849753859.png)
Now we have 4 data error values, which are the distances of the 4 segments to the prediction function that you see.  We ++square the error value of each piece of data, which is the **quadratic cost**, and finally average the squared errors (sum and divide by the total number), which is the mean square error++.  Mathematically expressed as:
*cost = (1/n) \* Σ(y - y_hat)²*
where n is the total number of data, and we have already covered the rest.  Sometimes, to save computation, ++the averaging step is omitted++ (*\*(1/n)*), and as long as the averaging step is omitted for each training session, the final training result will definitely be unaffected; of course, to facilitate derivation, ++sometimes it is also written as dividing by 2++, because we know that the derivative of x² is 2x, and 2 multiplied by 1/2 is reduced to 1, which is convenient for calculation.
The mean squared error MSE, which requires us to continuously reduce this error value, is the most common cost function we use when we first learn about neural networks.

The js code for MSE is:
```javascript
function MSE(out, out_hat){
    err = 0
    for(i=0;i<=out.length-1;i++){
        err += (out_hat[i] - out[i]) * (out_hat[i] - out[i]);
    }
    return err / out.length
}
console.log(MSE([1,2],[2,2]))//Output 0.5
//[1,2] is 2 true values, [2,2] is 2 predicted values corresponding to them
```

c++ version:
```cpp
#include<iostream>
#include<vector>
#include<cmath>

double MSE(std::vector<double> out, std::vector<double> out_hat){
    double err = 0;
    for(int i=0; i<out.size(); i++){
        err += pow(out_hat[i] - out[i], 2);
    }
    return err / out.size();
}

int main(){
    std::vector<double> out{1, 2};
    std::vector<double> out_hat{2, 2};
    std::cout<<MSE(out, out_hat)<<std::endl;//Output 0.5
    return 0;
}
//The principle is the same as the js version
```

2. **Maximum Likelihood Estimate (MLE)**
Many people talk about Gaussian distribution directly here, which scares many people off the bat.  In fact, there is no need for us to put these more difficult things in front of us first.  Let's break down the six words "Maximum Likelihood Estimate" one by one.  "Maximum", it seems that unlike the previous MSE cost function, here we are trying to find the maximum value of the MLE cost function, not the minimum value, why is that? Think about it carefully, when it comes to the maximum value, the first thing that should come to mind is probability.  The higher the probability, the greater the likelihood that it will happen.  As we continue reading, we see the words "likelihood" and "estimate".  Maybe we don't know what "likelihood" is yet, but we see the word "estimate" and we can be sure that this thing must have something to do with probability.  At this point, we should talk about **likelihood (L)**.

We learned in elementary school that **probability (P)** represents the likelihood of an event occurring, for example, P(A) represents the likelihood of event A occurring, where A might be something like "tossing a coin heads up", "winning the first prize", or "winning second place in a running race".  But when we are asking for a probability, we usually want to give a background condition, such as "the coin is homogeneous", "drawing from 10 tags, only 1 tag corresponds to the first prize", or "this contestant usually runs first half the time and second half the time", so that we can calculate that the probabilities corresponding to these three events are 1/2, 1/10 and 1/2, respectively.
However, the likelihood value, which we are learning today, is exactly the opposite of probability.  Now, I'm telling you that there are three kinds of lottery machines: the first one has 10 tags each time, 5 of which are first prize tags; the second one has 10 tags each time, 1 of which is a first prize tag; the third one has 10 tags each time, none of which are first prize tags.  I drew 100 times and won 8 times, so what lottery machine am I drawing from (without changing in between)?  You're more likely to answer: the second one.  This is because the second lottery machine, under the current background or current situation of drawing 8 prizes out of 100 draws, is more likely to fit this condition, or is more likely to be using this lottery machine.  In statistics, we call this: greater likelihood.  Just now we put aside all the obscure formulas, we should already know that likelihood is probably ++expressing what the background of an event is most likely to be after it has already happened++.
Now look back at the formula and there should be no more confusion.
Suppose we don't know if a coin is homogeneous, and we don't consider any less influential physical phenomena.  So we set the percentage of heads weight of the coin as θ (read as Theta), and the opposite is 1-θ.  We need to find the maximum likelihood value of L(θ), because we have already said that the maximum likelihood value means that the possibility of this θ being correct is the greatest.  So we start tossing the coin, and after tossing it 10 times, 2 times the head is down and 8 times it is up.  At this point, the probability of tossing the coin twice with both heads down is θ squared (because the probability of tossing the coin once with the head down is θ, and the simultaneous occurrence of two events is expressed as the multiplication of probabilities or likelihoods), and the probability of tossing the coin 8 times with all tails down is (1-θ) to the power of 8, and both of these events occurred simultaneously in our experiment, so we multiply again, and finally get
*L(θ) = (θ^2) * \[(1-θ)^8\]* 
At this point, we only need to find the maximum value of L(θ) to find the best parameter θ.

In neural networks, when the true value of our data is xi and the predicted value of the neural network is yi, we can use
*likelihood = P(xi|yi)*
to express it.  Where P() is the probability, the "|" symbol indicates conditional probability, P(A|B) means the probability of A occurring under the condition of B.  In our neural network, we are essentially calculating P(xi|NN), where NN is the set of all parameters (w and b) in the neural network.  P(xi|NN) can be understood as the probability of judging the result to be xi under the background of such a neural network model calculation (remember that xi is the true value, so the higher the probability, the closer it is to the true value, the better). But obviously we can't directly expand NN and calculate them one by one, but we know that NN finally outputs the value of yi, so we have the above 
*likelihood = P(xi|yi)*.
Expanding it using the likelihood value calculation method we just learned, we get
*cost = (yi^xi) * \[(1-yi)^(1-xi)\]*
However, this is generally used for models where the output value is a probability/component between 0-1.
When there are multiple training data, the cost can be written as:
*TotalCost = ∏i (yi^xi) * \[(1-yi)^(1-xi)\]*
where the "∏" symbol represents continuous multiplication, i.e., ++multiplying each of the final calculated results together++, because this is likelihood, and like probability, simultaneous occurrence is expressed as continuous multiplication. Finally, our goal becomes to find the maximum value of TotalCost.

Note! ++Maximum likelihood estimation is only applicable to probabilistic classification or fitting models where the output results are between 0 and 1, and this loss/cost function does not support predicted or true values with inputs less than 0 or greater than 1!++

I think it would be perfect to pair it with an interactive demo now.  Please open the Geogebra demo I made:
https://www.geogebra.org/m/wmfrg8r7
You will see an orange curve whose expression is:
![ ](./images/1696401537819.png)
xi is the true value of the data, p is the predicted value.  You can see an orange slider in the upper left corner of this demo page, sliding it left and right to adjust the value of xi, and at this point, observe the image and you will find that the value of the horizontal coordinate corresponding to the highest point of the orange function curve is the value of xi. Now pause and quickly answer this question: what do the horizontal and vertical axes of the function image corresponding to the orange function correspond to what we have talked about?
Good, it's not hard to understand, the vertical axis is obviously the likelihood value, since p is the predicted value, the p corresponding to the highest point of likelihood must be the p closest to the true value xi, so the horizontal axis must correspond to the predicted value p.
![ ](./images/1696402047670.jpg)
This is the maximum likelihood estimate.  Sometimes logarithms are used to avoid too many consecutive multiplications and powers, although I personally think that the power notation makes it easier to think of the essence of the likelihood value.

The js code for MLE is:
```javascript
function MLE(out, out_hat){
    likelihood = 1
    for(i=0;i<=out.length-1;i++){
        likelihood *= Math.pow(out_hat[i], out[i]) * Math.pow(1 - out_hat[i], 1 - out[i]);
    }
    return likelihood
}
console.log(MLE([0,1],[0.3,0.8]))//The output value is approximately 0.56
//[0,1] are the true values, [0.3,0.8] are the predicted values, both must be between 0 and 1
```

c++ version:
```cpp
#include <iostream>
#include <cmath>

using namespace std;

double MLE(int out[], double out_hat[], int n) {
    double likelihood = 1.0;
    for (int i = 0; i < n; i++) {
        likelihood *= pow(out_hat[i], out[i]) * pow(1 - out_hat[i], 1 - out[i]);
    }
    return likelihood;
}

int main() {
    int out[] = {0, 1};
    double out_hat[] = {0.3, 0.8};
    double result = MLE(out, out_hat, 2);
    cout << result << endl; //Output is 0.56
    return 0;
}
```

(Optional:) Finally, if you know about Gaussian/normal distributions, sometimes you can use the same logic to plug xi and yi into the normal distribution respectively.
https://www.geogebra.org/m/rkx8fy5z
You can look at this demo of mine, drag the slider of parameter b to make the shape of the red parabola closer and closer to the green parabola, then the magnitude of the likelihood value represented by the blue straight line will start to increase until the two parabolas completely coincide and the likelihood value reaches its highest point.  You can go to the algebraic section of the demo and study the principle yourself, I will not explain it too much here.
![ ](./images/1696408303584.png)
There is also a part of the code related to this, I only wrote the js version, because this thing is not very commonly used in MLE, you can understand it briefly (the gradient descent method involved, it is recommended that you can selectively read this code after we finish learning the gradient descent method):
```javascript
function calculateLikelihood(data) {
  let likelihood = 1;
  //Initialize parameter b
  let b = 2;
  sqrt2pi = Math.sqrt(2 * Math.PI);
  //Loop update parameter b
  for (i = 0; i <= 20; i++) {
    sigma = 1
    //Calculate the current likelihood function value
    currLikelihood = 1;
    for (let j = 0; j < data.length; j++) {
      [x, y] = data[j];
      yhat = Number(Math.pow(2,x)+b);
      error = y - yhat;
      
      currLikelihood *= (1 / (sqrt2pi * sigma)) * Math.exp(-0.5 * (error / sigma) ** 2);
    }
    
    console.error("Likelihood value: "+currLikelihood)
    
    function normalDistributionDerivative(sigma, err) {
        coefficient = 1 / (Math.sqrt(2 * Math.PI) * sigma);
        exponent = (-1 / 2) * Math.pow((err / sigma), 2);
        derivative = coefficient * Math.pow(Math.E, exponent) * (-1 / sigma) * err;
        return derivative;
    }
    
    d = 0
    
    for (let j = 0; j < data.length; j++) {
        [x, y] = data[j];
        yhat = Number(Math.pow(2,x)+b);
        error = y - yhat;
        d += normalDistributionDerivative(1, error)
    }
    
    d = d / data.length
    b -= rate * d
    log("b ← "+b)
  }
  return b;
}
rate = 1
data = [[-7.6, 0], [-0.85, 0.55], [1.5, 2.83]]
console.info("Final result:"+calculateLikelihood(data))
```

### 3.2 Gradient Descent

Gradient descent is actually a very wonderful and ingenious method in neural networks, and it's amazing to think about.
Let's put aside the formulas for a moment and take a step-by-step look at how this method came about.

Before studying this section, make sure you understand what a **derivative** is.  If you don't, I recommend you watch 3Blue1Brown's videos, the first few videos in his Calculus series explain derivatives in detail, and they're very well done.  The video link is mentioned at the beginning of the book.

First, let's say there is a mountain range, and we are now standing somewhere in the mountain, and the fog in the mountain is so heavy that we can only see what the slope of the land is under our feet.  The situation is now like this:
![ ](./images/1696411150874.png)

Now, we need to get to the bottom of the mountain as soon as possible to find a village to spend the night.  Just looking at this picture, answer me with common sense: should we go to the left or right of the picture?  It should be the left, but why?  After all, we can't see in which direction the bottom of the mountain is, we can only see the slope of the land under our feet. Obviously, we are using the fact that the ground under our feet is sloping to the left to infer that the left is probably the way down the mountain.  So, let's say that the horizontal coordinate of our current location (x is smaller on the left and x is larger on the right in the figure) is x, the height, i.e., the vertical coordinate, is y, and the derivative (slope of the tangent) of the land directly in front of our feet is d (don't worry about how to find the specific value of this d for now).  Then when d is greater than 0, we know that the land is sloping to the left, and to reach the bottom of the mountain, we need to go to the left, i.e., subtract a distance from x (x←x-△x and △x>0 (the "←" symbol indicates assigning the value of a variable to the content to the right of the arrow; the △ symbol usually indicates the number added, i.e., the increment)); conversely, when d is less than 0, we know that the land is sloping to the right, and to reach the bottom of the mountain, we need to go to the right, i.e., add a distance from x (x←x+△x and △x>0).  We've been stipulating that △x must be greater than 0, and we still have to discuss by category, which is too much trouble, and we also need to know how far to go (i.e., what △x is appropriate).  If we keep going a long way without regard to the current slope of the land under our feet, we're likely to miss the lowest point of the mountain; conversely, if △x is too small, it may take a long time to reach the lowest point.  At this point, we noticed the derivative d we just talked about.  We know that the larger the d, the more the tangent slopes to the left, and the smaller the d, the more the tangent slopes to the right.  Moreover, we know that when the slope is very large, it means that the direction must be down the mountain very quickly, so moving quickly in this direction will allow us to reach the bottom of the mountain at the fastest speed; conversely, when the slope is very small, we may be close to the bottom of the mountain (maybe not sometimes, but we'll talk about that later), we need to slow down to make sure we don't miss the lowest point.  Therefore, we arrive at the following formula:
*x←x-d*
This way, we can ++decide how far to move in which direction based on the magnitude of the slope++. In order to better ++control the speed, we also add a **hyperparameter** called the **learning rate (η)**++, ++multiplying the derivative d by it so that the learning rate can control our learning speed in real time and manually++, resulting in the formula:
*x←x-η\*d*
The reason why it is called a hyperparameter is to avoid confusion with the parameters of the neural network itself (i.e., w and b).
Later on, we will also learn about some hyperparameters, and it is very important to adjust them.  ++If the learning rate is too large, we are likely to artificially miss the lowest point, and if it is too small, it will waste a lot of computational resources.++
Now it's time to experience the demo:
https://www.geogebra.org/m/gequxauv
Open this demo I made, in this demo, b is the horizontal coordinate of the image, we need to find the appropriate b, which is the lowest point of the curve, we need to go down step by step.
![ ](./images/1696426188115.png)
First let point A come to this position, you can imagine it as a ball, it is now going to roll down the hillside. In the slider in the upper left corner of the page, adjust the learning rate η to about 0.1, then click on the purple train button and observe how A rolls down to the bottom step by step.  The slope of the black tangent here is the derivative d, which I use k to represent in this demo. After about 50 clicks on the train button, A has rolled to the bottom.
![ ](./images/1696426536077.png)
The blue dots record the trajectory of A's movement, and you can see that when the slope is steeper and the slope is larger, A rolls down quickly, and when it reaches the low point, the slope becomes gentler, the slope decreases, and A slowly stops until it has almost reached the lowest point.
Now, first adjust the learning rate to 0.05, find the play button in the lower left corner of the page, click it and it will start training in real time. Drag point A to the hillside and you will see it start to roll down on its own.  We change the learning rate to 0.01 and it will roll down much slower; we adjust the learning rate to 1 and it will always oscillate on both sides of the valley, always missing the lowest point and unable to converge.  Of course, no one defines what the learning rate must be, you should consider the magnitude of the learning rate based on the actual situation during training, it may be 0.0001, it may be 20, etc., and we will also learn about adaptive learning rate mechanisms later on.

Now we're getting pretty close to gradient descent, we already know how to find the lowest point of a function.  Remember what our ultimate goal was in the last section?  We want to find the smallest loss value so that our neural network can better fit or divide the data points, and we need to optimize and adjust the weights and bias parameters of the neural network to achieve this.  Therefore, in the example we just had, the horizontal coordinate x actually corresponds to a parameter we want to optimize in the neural network, while the vertical coordinate (height) is the loss value calculated by the cost function.  It is important to note here that the MSE cost function needs to find the minimum value, while the MLE cost function needs to find the maximum value (i.e., gradient ascent). However, there are many weight and bias parameters in a neural network, so a two-dimensional coordinate system is not enough at this point.  Each weight and bias parameter here is an independent variable, and the dependent variable is the cost/loss value (so the dimensionality here is very large and it is impossible to visualize it completely), so in the process of finding the minimum value of the cost function, it is not just moving left and right.  At this point, the concept of **gradient** should come out.

First of all, let's be clear that ++the direction of movement in the process of finding the minimum value of the cost function is a vector++, which has both direction and length. To simplify the problem, let's first assume that we have only one neuron, and its expression is: *out = wx + b*. Then, assuming that we have enough training data, we draw such a function image of the loss value with respect to w and b:
![ ](./images/1696473928655.png)
In the image, the small white ball is the loss value corresponding to the parameters w and b of the neuron, and now we need to adjust w and b to minimize the loss value.  So, based on what we just learned, we first draw a vector, note that ++this vector points in the direction of the fastest descent++, not to the lowest point!  This vector should be decomposed onto the w and b axes, because they are the parameters we want to adjust (while the vertical axis loss is not directly adjustable):
![ ](./images/1696474777106.png)
In fact, the decomposition of the vector here can be directly understood as projecting a three-dimensional direction onto a two-dimensional plane, which is a projection, not a tiling, so the length will change, which is equivalent to deleting the vertical axis dimension of the vector.  This two-dimensional vector in this example is basically what we call the gradient (++when there are n parameters to optimize, the gradient will have n dimensions, because it does not include the dimension of the dependent variable loss++).

However, we can't get this gradient/vector directly, we ++need to calculate its components in the w and b directions, i.e. the partial derivatives++.
![ ](./images/1696484033134.png)
As shown in the figure, we take the position of the current parameters w and b as the origin, and decompose the gradient vector onto the w and b directions.

If you are not familiar with partial derivatives, I will briefly introduce them:
We know that the derivative is the slope of the tangent, so in more than two dimensions, one dependent variable may be influenced by two independent variables at the same time.  Simply put, in order to determine the influence of a certain independent variable on the dependent variable, we first treat the other variables as constants (in the actual neural network training process, we know the specific values of these variables at this time, such as w and b, we just want to know how to adjust and optimize and update them), at this time only one independent variable is left, and then we can find the derivative.  At this time, the derivative obtained is called "the partial derivative of <dependent variable> with respect to <independent variable not treated as a constant>", denoted as *∂dependent variable/∂independent variable*, where the "∂" symbol is read as round.  Here is an example:
![ ](./images/1696485570735.png)
Suppose the red axis is the x-axis, the green axis is the y-axis (it may be blocked), and the blue axis is the z-axis.  The blue surface is the graph of z as a function of x and y. Point A is on this function surface, and its x, y coordinates are (2,1).  At this point, to find the partial derivative of z with respect to y at the coordinates of point A, simply draw a plane along this point (i.e., the plane x=2), cut the function surface along the y-axis direction, so that the function surface forms a new curve on the plane, and then find the derivative on the curve, at this time, the changing quantities on the curve are only z and y, and x is treated as a constant 2.  At this point, you can find the partial derivative *∂z/∂y*.
The partial derivative of z with respect to x is the same:
![ ](./images/1696486453856.png)
I won't go into too much detail.

Back to the gradient descent method, we have two components in the w and b directions, which can be written as two partial derivatives, namely *∂loss/∂w* and *∂loss/∂b*, respectively.  The length of each component just now is equal to the magnitude of the corresponding partial derivative.  Without considering how to calculate the magnitude of these two partial derivatives specifically, let's go back to the overall goal, now we can easily define the gradient.  ++The symbol for the gradient is "▽", read as nabla.  When we talk about the gradient, we are generally referring to all these partial derivatives, usually written as ▽C, which is an abbreviation for ▽Cost (the gradient of the cost function). Usually when we talk about the gradient of a variable, we are talking about the partial derivative of cost or loss with respect to it.  In the whole neural network, there are many parameters w and b, so each of them has its own gradient++.

When we want to optimize a parameter w or b by gradient descent, we can update it like this:
*n←n-η\*▽n*
Of course, when using the maximum likelihood estimate MLE as the cost function, it should be:
*n←n+η\*▽n*
For example, to update a parameter w5,
*w5←w5-η\*▽w5*
When training our neural network, ++every weight and bias parameter of every neuron in every layer needs to be updated this way++ to achieve the smallest error value.

Since we haven't learned how to accurately calculate the specific values of the gradient, in order to experience the gradient descent method first, let's use the most basic method of derivation, which is to give a small increment, and in our code, this increment is infinitesimally small.
In the following example code, there is only one sigmoid neuron, and we will update w and b in turn by gradient descent until the error meets our requirements.
The js code for the gradient descent method is:
```javascript
w = 1
b = 0

data = [0.4,0.9]//Training data
rate = 0.6//Learning rate
train(data)

//S-type neuron
function neuron(x){
    return 1 / (1 + Math.exp(-(w*x+b)))
}

//Mean square error cost function
function MSE(out, out_hat){
    return (out - out_hat) * (out - out_hat) / 2
}

//Training function
function train(data){
    h = 1e-10//Derive small increments
    i = 0//Training Count
    while(true){
        predict0 = neuron(data[0])
        cost0 = MSE(data[1], predict0)
        console.log("Training Count: "+i+"\nMSE: "+cost0+"\n----------")
        if(cost0<=0.001){//Train until the mean square error is less than 0.001
            console.log("Training completed")
            break;
        }
        
        //Calculate gradients separately
        w += h
        predict1 = neuron(data[0])
        w -= h
        cost1 = MSE(data[1], predict1)
        gradient_w = (cost1 - cost0) / h
        
        b += h
        predict1 = neuron(data[0])
        b -= h
        cost1 = MSE(data[1], predict1)
        gradient_b = (cost1 - cost0) / h
        
        //Update weights and biases separately
        w -= rate * gradient_w
        b -= rate * gradient_b
        
        i++
    }
}
```

c++ version:
```cpp
#include <iostream>
#include <cmath>

double w = 1;
double b = 0;

double neuron(double x){
    return 1 / (1 + exp(-(w*x+b)));
}

double MSE(double out, double out_hat){
    return (out - out_hat) * (out - out_hat) / 2;
}

void train(double data[]){
    double rate = 0.6;
    double h = 1e-10;
    int i = 0;
    
    while(true){
        double predict0 = neuron(data[0]);
        double cost0 = MSE(data[1], predict0);
        
        std::cout << "Training Count: " << i << "\nMSE: " << cost0 << "\n----------" << std::endl;

        if(cost0 <= 0.001){
            std::cout << "Training completed" << std::endl;
            break;
        }

        w += h;
        double predict1 = neuron(data[0]);
        w -= h;
        double cost1 = MSE(data[1], predict1);
        double gradient_w = (cost1 - cost0) / h;
		
        b += h;
        predict1 = neuron(data[0]);
        b -= h;
        cost1 = MSE(data[1], predict1);
        double gradient_b = (cost1 - cost0) / h;
		
        w -= rate * gradient_w;
        b -= rate * gradient_b;
        i++;
    }
}

int main(){
    double data[] = {0.4, 0.9};
    train(data);
    return 0;
}
//The principle comments can be found in the JS version above
```

In the next section, we will continue to learn how to use the backpropagation algorithm to calculate the gradient value.

### 3.3 Backpropagation Algorithm

Before you learn this section, make sure you understand **the product and chain rules for derivation**.  If you don't, I recommend you watch 3Blue1Brown's videos, which are linked at the beginning of the book.

We already know that the gradient of a parameter is the partial derivative of the cost with respect to it.  The process of backpropagation is to calculate this partial derivative by the chain rule.  Here I'm just going to give a definition first to make it easier to explain later, and it doesn't matter if you don't understand it because we're going to learn the backpropagation algorithm in a very clever way, rather than just applying the formulas, because if you do, you'll completely miss the real principle of backpropagation.
In my experience, if something is hard to understand, then giving an example is the best way to go. Here's a neural network we're already very familiar with:
![ ](./images/1696678707481.png)
I won't explain the structure any further.  When the network outputs a predicted value of a6 (representing the activation value of the 6th neuron in the figure), a cost value is calculated using MSE, and we need to adjust some parameters w and b to reduce this value. Let's simplify the problem a bit, let's not look at neural networks and neurons, let's just write a simple function of y with respect to x: y=w\*x, assuming that when I input x=1, it outputs y=2, and I want it to output y=3, now our sole purpose is to keep its output value approaching 3, we'll transform the problem again, calculate the loss value (without using any loss function to simplify) loss=3-2=1, (usually people are used to writing the true value minus the predicted value this way) so our purpose becomes to keep trying to reduce the size of loss.  From here, let's see how to backpropagate this error value.
Now the error value is 1, in order to reduce it, we have two options: the first is to reduce the true value of 3 a little, the error will of course be reduced, but this is obviously not feasible, otherwise the true value can still be called the true value? Then we can only adjust the size of the predicted value. At this point, the error is backpropagated from loss to the predicted value.
So, our current phase goal is to make the predicted value a little larger in order to reduce the loss value.  In order to increase the predicted value, we again have two options: the first is to increase the size of w, and the second is to increase the size of x.  The latter is obviously not feasible, because it is an input value, and like the true value, it cannot be modified.  So we understand that to reduce the loss value, we must increase w.  As for how much to increase, it actually depends on the size of x.  Please think about it carefully for yourself, why is this?
Actually, it's not a good idea to use this formula to understand it at this point, so let's expand it a bit: y=w1\*x1+w2\*x2, at this point, let's say x1=0.2, x2=0.9, and now we want to make the y value bigger as soon as possible, and I'm only allowing you to increase either w1 or w2 by 0.5, which one would you choose to increase first? w2 of course, in fact, we can calculate that if we increase w1 by 0.5, then y will increase by 0.5\*x1=0.1, and if we increase w2 by 0.5, y will increase by 0.5\*x2=0.45. In fact, you can think of it as both parameters w have their own ++responsibility++ in increasing the value of y, but because the coefficient multiplied by w1 is smaller (because x cannot be optimized, so I treat it as a constant, i.e., the coefficient of the corresponding w), its responsibility is directly multiplied by 0.2 for efficiency, while the responsibility of w2 is multiplied by 0.9 for efficiency, so in order to increase the overall optimization efficiency, we will ++assign more responsibility++ to w2.
So, when y has err this much error to be modified, we determine: w1 should take err\*x1 of the responsibility, w2 should take err\*x2 of the responsibility, this way, we only need to use the gradient descent method again, over and over again to update w1 and w2 (remember to multiply by the appropriate learning rate) to continuously reduce the error loss value.
I wonder if you have noticed that we have inadvertently completed a partial derivative and a ++chain derivation++, i.e. the partial derivative of loss with respect to w1 and the partial derivative of loss with respect to w2.  The formula for the chain rule of derivation is written out as follows (here I have to use y_hat to represent the predicted value in order to distinguish it from the true value):
*∂loss/∂w1 = (∂loss/∂y_hat) \* (∂y_hat/∂w1)*
*∂loss/∂w2 = (∂loss/∂y_hat) \* (∂y_hat/∂w2)*
If you are familiar enough with the chain rule and the product rule of derivation, you will find that the calculation process of these formulas is completely consistent with our logical reasoning process just now.
According to the rule, the ∂loss/∂y_hat just now is equal to -1 (in loss=y-y_hat, the coefficient of y_hat is -1, and according to the product rule of derivation, the partial derivative is equal to this -1), and ∂y_hat/∂w1 is equal to x1, so we finally get: ∂loss/∂w1 = -1 \* x1 = -x1 using the chain rule.
Having said that, it is obvious that we can now go back and take a look at the neural network.

If we use MSE as the cost function, according to the derivation of the power function, we know that the derivative of the square of a number is equal to 2 times that number, for example, the derivative of x^2 is equal to 2x. In backpropagation, we can use only one training data at a time (++note that in the normal gradient descent method, we forward propagate all the training data separately to get a bunch of corresponding predicted values, and then backpropagate to get the gradient values of each weight and bias, during which time we do not adjust any parameters of the neural network, but save the gradient values of each weight and bias obtained by each backpropagation, after all the data has been propagated forward and backward, for each weight and bias parameter, we have several gradient values of it, which is the gradient value calculated by backpropagating each training data, we add them up and take the average, as the final gradient involved in optimizing this parameter, we will also learn a more efficient gradient descent method in the next section Please do not confuse them++), so the cost or loss value can be written as:
*(1/2) \* (y-y_hat)^2*
When deriving, multiplying by 2 and one-half cancels out, so its derivative is y-y_hat, and taking the partial derivative of y-y_hat with respect to y_hat gives -1, and then chaining the derivative, we finally get that the partial derivative of the MSE value with respect to y_hat is -1*(y-y_hat), which is y_hat-y, perfect.
We continue to backpropagate the error, assuming that the neurons all use the Sigmoid activation function (abbreviated as S(n) later on), then the derivative of its output activation value S(n) is *S(n)\*(1/S(n))*, you can study its proof yourself if you are interested, but it is not the focus of today, we still have to backpropagate the error.
The n in the formula just now can be expanded as ∑i wi\*xi+b, at this point we already see the first batch of parameters that need to be updated, the partial derivative of each w is the value of its corresponding x, and the partial derivative of b is 1, because its coefficient is 1.
For example, suppose that in a certain neuron in the output layer there is a weight parameter w8 and a bias parameter b6, then their gradients are:
*▽w8 
 = ∂loss/∂w8
 = (∂loss/∂y_hat) \* (∂y_hat/∂n) \* (∂n/∂w8)
 = (y_hat-y) \* S(n)\*(1-S(n)) \* x8*

*▽b6 
= ∂loss/∂b6
= (∂loss/∂y_hat) \* (∂y_hat/∂n) \* (∂n/∂b6)
= (y_hat-y) \* S(n)\*(1-S(n)) \* 1*

This way, we can ++calculate the gradients of all the parameters in the hidden layer++. However, ++each x in ∑i wi\*xi+b can actually be modified indirectly++ because x here is the output value of the neuron in the previous layer, and ++there are also some w and b parameters on that neuron that need to calculate the gradient++, so we can ++continue to backpropagate the error to the previous layer++ (hidden layer).
For this hidden layer, let's take neuron #3 in the figure above as an example, whose output value is forward propagated to neuron #6, so neuron #6's error is also partially backpropagated to neuron #3.  At this time, it is very simple to use the formula to express the gradient of each parameter of neuron 3.  Suppose the weight connecting neuron 3 and neuron 6 is w36, the weight connecting neuron 3 and input layer neuron x1 is w13, the activation value of neuron 3 is a3, the value before neuron 6 is activated (∑i wi\*xi+b) is n6, and the value before neuron 3 is activated is n3, then the gradient of w13 is:
*▽w13
= ∂loss/∂w13
= (∂loss/∂a3) \* (∂a3/∂w13)
= (∂loss/∂y_hat) \* (∂y_hat/∂n) \* (∂n6/∂a3) \* (∂a3/∂n3) \* (∂n3/∂w13)
= (y_hat-y) \* S(n6)\*(1-S(n6)) \* w36 \* S(n3)\*(1-S(n3)) \* x1*

Finally, ++if there are multiple output neurons, then during backpropagation the previous hidden layer will call the average of the error gradients propagated from these neurons++, which can be understood as each output neuron having its own idea of how to change the parameters of the previous layer neurons, then taking the average will do.  We will explain this in detail in the next section.

++Neural networks trained using the backpropagation algorithm can be called **Back Propagation Neural Networks (BPNN)**++.

I think this should be enough to help us understand the backpropagation algorithm. But now we are not able to write the complete code yet, we need to learn about "stochastic gradient descent" first.

### 3.4 Stochastic Gradient Descent

In learning the previous two sections, you may have noticed a problem: the neural network is not just adjusting the parameters to fit one training data, but to fit a lot of data at the same time and to predict the new data as accurately as possible. So, for each training data, there will be an error loss cost value, and each data will have its own idea on how to adjust the parameters on each neuron.  So how to adjust these parameters to satisfy the requirements of each data at the same time? Or, for a certain parameter w or b, there are n pieces of training data, and each piece of training data will have its own gradient for w and b after backpropagation to w or b.  For example, the first piece of training data backpropagates the error to w and gets ▽w\[1\], and the second piece of training data gets ▽w\[2\] after propagation, then how to optimize w now?
According to the traditional normal gradient descent method, we should calculate the average value of these gradients, then the way to update w is:
*w←w-(∑t ▽w\[t\])/n*
where t represents the t-th training data used to get the gradient, and n is the total number of data.
But we will quickly realize that this is completely unenforceable.  For example, one piece of data wants to make w a little smaller, another piece of data wants to make it a little bigger, the gradients cancel each other out because one is positive and the other is negative, and finally it's like taking an average, not fitting any of the data, just reducing the error to some extent (which can be understood as falling into a local optimum (low) point of the cost function).
So, the **Stochastic Gradient Descent (SGD)** came along. This time, we ++use only 1 data for each training (descent)++, we throw a training data input value into the neural network, propagate forward to get the predicted value, and then backpropagate the loss value of this one data to get the gradient of each weight and bias, this time we will not save these gradients, but calculate the gradient of a parameter and immediately update and optimize this parameter until every parameter of every neuron in each layer is traversed and updated in this way.  After training this one data, we go and find another data to train, and so on. This way, ++we train each sample, that is, all the data in the training data set, individually++, which is expressed in the formula as follows:
*Loop through:
w←w-▽w\[t\]*

Compare the difference between the two gradient descent training methods, if you draw a loss-train image (the change of loss value with the increase of training data), the two gradient descent training statistics after the image is probably similar to the following figure:
![ ](./images/1697552067655.png)
Blue is the training effect of stochastic gradient descent, and red is the normal gradient descent.  There are a total of 4 pieces of data to be trained, and in normal gradient descent, one training (descent) will update and optimize the parameters based on 4 pieces of data at the same time, while in SGD, each gradient descent will only optimize one piece of data, so it appears to be very jittery and unstable.  But as it turns out, SGD can still reduce the loss value better (i.e., converge) despite this.

SGD makes large-scale training of neural networks possible in theory. Now, we can give the complete code for training and predicting a complete neural network.


## Section 4 Complete Code

Note: e in the js and c ++codes represents Scientific notation. For example, 1e-5 represents 1 times 10 to the power of -5. Please note that it is separate from the constant e, which has nothing to do with logarithms.

JS version:
```javascript
//Randomly initialize neural network parameters

network = [

    //A hidden layer with 4 neurons

    [

        [[0.1,0.7],0.1],

        [[0.4,0.8],0.9],

        [[0.7,0.1],0.6],

        [[0.1,0.4],0.5]

    ],

    //Output layer 2 neurons

    [

        [[0.3,0.4,0.5,0.2],0.6],

        [[0.3,0.4,0.5,0.2],0.6],

    ]

]

//Temporarily store the current activation values of all neurons in the neural network for backpropagation calls

networkn = [

    [0,0,0,0],

    [0,0]

]

//Temporarily store the values of ∑ wx+b before all neurons in the neural network are activated for backpropagation calls

networkb = [

    [0,0,0,0],

    [0,0]

]

//A neuron

function Neuron(w, x, b) {

    //Use LeakyRelu as the activation function (will be learned later)

    function leakyRelu(x) {

        if (x >= 0) {

            return x;

        }else{

            return 0.1 * x;

        }

    }

    

    //Sum ∑ wx+b

    function sigma(w, x, b) {

        return w.reduce((acc, curr, i) => acc + curr * x[i], 0) + b;

    }

    

    sum = sigma(w, x, b)//Value before activation

    

    return [sum,leakyRelu(sum)];

}

//LeakyRelu's derivative for backpropagation calls

function leakyReluDerivative(x) {

  if (x >= 0) {

    return 1;

  }else{

    return 0.1;

  }

}

//Mean square error of individual data

function MSE(out, out_hat){

    return (out_hat - out) * (out_hat - out) / 2;

}

//Derivative of mean square error

function MSEDerivative(out, out_hat){

    return out_hat - out

}

//Prediction - Forward propagation

function predict(content){

    //Spread forward to the hidden layer

    for(m=0;m<=networkn[0].length-1;m++){

        r0 = Neuron(network[0][m][0],content,network[0][m][1])

        networkb[0][m] = r0[0]//Inactive value

        networkn[0][m] = r0[1]//Activation value

    }

    

    //Spread forward to the output layer

    for(n=0;n<=networkn[1].length-1;n++){

        r1 = Neuron(network[1][n][0],networkn[0],network[1][n][1])

        networkb[1][n] = r1[0]//Inactive value

        networkn[1][n] = r1[1]//Activation value

    }

    return networkn[1];

}

//Training backpropagation stochastic gradient descent

function trainNet(dt){

    out_hat = predict(dt[0])//Prediction

    MSError = 0

    //Calculate the MSE loss value to check the training effectiveness

    for(l=0;l<=out_hat.length-1;l++){

        MSError += MSE(dt[1][l], out_hat[l])

    }

    //Calculate the partial derivative of the inactive value of output layer neurons by multiplying the learning rate by the loss value for each neuron separately, reducing the subsequent computational workload

    rMEdN = []

    for(l=0;l<=out_hat.length-1;l++){

        rMEdN.push(rate * MSEDerivative(dt[1][l], out_hat[l]) * leakyReluDerivative(networkb[1][l]))

    }

    //Calculate the mean of each result above again, and use it as the desired mean for each output layer neuron to backpropagate how to adjust all parameters of the hidden layer

    rMEdNA = rMEdN.reduce((acc, curr) => acc + curr, 0) / rMEdN.length

    

    //Update output layer weights

    for(p=0;p<=networkn[1].length-1;p++){

        for(q=0;q<=network[1][p][0].length-1;q++){

            network[1][p][0][q] -= rMEdN[p] * networkn[0][q]

        }

    }

    

    //Update output layer bias

    for(p=0;p<=networkn[1].length-1;p++){

        network[1][p][1] -= rMEdN[p]

    }

    

    //Update hidden layer weights

    for(p=0;p<=networkn[0].length-1;p++){

        for(q=0;q<=network[0][p][0].length-1;q++){

            network[0][p][0][q] -= rMEdNA * network[1][0][0][p] * leakyReluDerivative(networkb[0][q]) * dt[0][q]

        }

    }

    

    //Update hidden layer bias

    for(p=0;p<=networkn[0].length-1;p++){

        network[0][p][1] -= rMEdNA * network[1][0][0][p] * leakyReluDerivative(networkb[0][p])

    }

    

    return MSError;

}

function train(dt){

    var start = Date.now()

    i = 0

    while(true){

        i++

        err = 0

    //Gradient descent updates and optimizes parameters for each training data

        for(c=0;c<=dt.length-1;c++){

            preErr = trainNet(dt[c])//Gradient descent once

            err += preErr

        }

        //Determine whether the loss value meets the requirement of being less than or equal to the target loss value

        if(err<=aim){

            var elapsed = Date.now() - start;//Training time statistics

            console.info("Training completed with err <= "+aim+" ("+err+")")

            console.log(">>> finished "+dt.length*i+" steps ("+i+" rounds) gradient descent in "+elapsed+"ms <<<")

            break;

        }else{

            console.error("Round: "+i+"  Training: "+dt.length*i+"  MSE: "+MSError)

        }

    }

}

rate = 0.17//Learning rate

aim = 1e-5//Target loss value

train([

    [[1,1],[0,0]],

    [[0,0],[1,1]],

    [[0,1],[1,0]],

    [[1,0],[0,1]],

])//Train four sets of training data

console.log(predict([1,0]))//The predicted result is close to 0.1
```

c++ version: 
```cpp
#include <iostream>

#include <vector>

#include <cmath>

using namespace std;

//Global variables

std::vector<std::vector<std::vector<std::vector<double>>>> network;//Due to the fact that C++vectors do not allow different types of data (i.e. vectors and doubles in C++) to be nested in each layer like arrays in JS, it is necessary to add a {} to the bias coefficient to make it also a vector. For details, please refer to the network assignment in the main function

std::vector<std::vector<double>> networkn;
//Temporarily store the current activation values of all neurons in the neural network for backpropagation calls

std::vector<std::vector<double>> networkb;
//Temporarily store the values of ∑ wx+b before all neurons in the neural network are activated for backpropagation calls

double rate;//Learning rate

double aim;//Target loss value

//A neuron

vector<double> neuron(std::vector<double> w, std::vector<double> x, double b) {

    //Use LeakyRelu as the activation function (will be learned later)

    auto leakyRelu = [](double x) {

        if (x >= 0) {

            return x;

        }

        else {

            return 0.1 * x;

        }

    };

    //Sum ∑ wx+b

    auto sigma = [&w, &x, b]() {

        double sum = 0;

        for (int i = 0; i < w.size(); ++i) {

            sum += w[i] * x[i];

        }

        return sum + b;

    };

    double sum = sigma();//Value before activation

    return { sum,leakyRelu(sum) };

}

//LeakyRelu's derivative for backpropagation calls

double leakyReluDerivative(double x) {

    if (x >= 0) {

        return 1;

    }

    else {

        return 0.1;

    }

}

//Mean square error of individual data

double MSE(double out, double out_hat) {

    return (out_hat - out) * (out_hat - out) / 2;

}

//Derivative of mean square error

double MSEDerivative(double out, double out_hat) {

    return out_hat - out;

}

//Prediction - Forward propagation

vector<double> predict(vector<double> content) {

    //Spread forward to the hidden layer

    for (int m = 0; m <= networkn[0].size()-1; m++) {

        auto r0 = neuron(network[0][m][0], content, network[0][m][1][0]);//Unlike the JS version, [0] needs to be added at the end for the reason stated in the global variable declaration

        networkb[0][m] = r0[0];//Inactive value

        networkn[0][m] = r0[1];//Activation value

    }

    //Spread forward to the output layer

    for (int n = 0; n <= networkn[1].size()-1; n++) {

        auto r1 = neuron(network[1][n][0], networkn[0], network[1][n][1][0]);

        networkb[1][n] = r1[0];//Inactive value

        networkn[1][n] = r1[1];//Activation value

    }

    return networkn[1];

}

//Training backpropagation stochastic gradient descent

double trainNet(vector<vector<double>> dt) {

    std::vector<double> out_hat = predict(dt[0]);//Prediction

    double MSError = 0;

    //Calculate the MSE loss value to check the training effectiveness

    for (int l = 0; l <= out_hat.size() - 1; l++) {

        MSError += MSE(dt[1][l], out_hat[l]);

    }

    //Calculate the partial derivative of the inactive value of output layer neurons by multiplying the learning rate by the loss value for each neuron separately, reducing the subsequent computational workload

    std::vector<double> rMEdN;

    for (int l = 0; l <= out_hat.size() - 1; l++) {

        rMEdN.push_back(rate * MSEDerivative(dt[1][l], out_hat[l]) * leakyReluDerivative(networkb[1][l]));

    }

    //Calculate the mean of each result above again, and use it as the desired mean for each output layer neuron to backpropagate how to adjust all parameters of the hidden layer

    double sum = 0;

    for (int i = 0; i < rMEdN.size(); i++) {

        sum += rMEdN[i];

    }

    double rMEdNA = 0;

    if (rMEdN.size() > 0) {

        rMEdNA = sum / rMEdN.size();

    }

    //Update output layer weights

    for (int p = 0; p <= networkn[1].size() - 1; p++) {

        for (int q = 0; q <= network[1][p][0].size() - 1; q++) {

            network[1][p][0][q] -= rMEdN[p] * networkn[0][q];

        }

    }

    //Update output layer bias

    for (int p = 0; p <= networkn[1].size() - 1; p++) {

        network[1][p][1][0] -= rMEdN[p];

    }

    //Update hidden layer weights

    for (int p = 0; p <= networkn[0].size() - 1; p++) {

        for (int q = 0; q <= network[0][p][0].size() - 1; q++) {

            double averagenN = 0;

            for (int s = 0; s <= network[1].size() - 1; s++) {

                averagenN += network[1][s][0][p];

            }

            averagenN = averagenN / network[1].size();

            network[0][p][0][q] -= rMEdNA * averagenN * leakyReluDerivative(networkb[0][q]) * dt[0][q];

        }

    }

    //Update hidden layer bias

    for (int p = 0; p <= networkn[0].size() - 1; p++) {

        double averagenN = 0;

        for (int s = 0; s <= network[1].size() - 1; s++) {

            averagenN += network[1][s][0][p];

        }

        averagenN = averagenN / network[1].size();

        network[0][p][1][0] -= rMEdNA * averagenN * leakyReluDerivative(networkb[0][p]);

    }

    return MSError;

}

void train(vector<vector<vector<double>>> dt) {

    // start = Date.now()

    int i = 0;

    while (true) {

        i++;

        double err = 0;

        //Gradient descent updates and optimizes parameters for each training data

            for (int c = 0; c <= dt.size() - 1; c++) {

                double preErr = trainNet(dt[c]);//Gradient descent once

                err += preErr;

            }

            //Determine whether the loss value meets the requirement of being less than or equal to the target loss value

            if (err <= aim) {

                //var elapsed = Date.now() - start;// Training time statistics

                std::cout << "Training completed with err <= " << aim << " (" << err << ")" << std::endl;

                std::cout << ">>> finished " << dt.size() * i << " steps (" << i << " rounds) gradient descent in " << /*elapsed + */"ms <<<" << std::endl;

                break;

            }

            else {

                std::cout << "Round: " << i << "  Training: " << dt.size() * i << "  loss: " << err << std::endl;

            }

        }

}

int main() {

    //Assign value

    network = {

        {

            {{0.1, 0.7}, {0.1}},

            {{0.4, 0.8}, {0.9}},

            {{0.7, 0.1}, {0.6}},

            {{0.1, 0.4}, {0.5}}

        },

        {

            {{0.3, 0.4, 0.5, 0.2}, {0.6}},

            {{0.3, 0.4, 0.5, 0.2}, {0.6}}

        }

    };//The definition method of bias coefficient is different from the JS version, and the reason can be found in the global variable declaration

    networkn = {

        {0, 0, 0, 0},

        {0, 0}

    };

    networkb = {

        {0, 0, 0, 0},

        {0, 0}

    };

    rate = 0.17;//Learning rate

    aim = 1e-10;//Target loss value

    train({

        { {1, 1} ,{0, 0}},

        { {0, 0} ,{1, 1}},

        { {0, 1} ,{1, 0}},

        { {1, 0} ,{0, 1}},

    });//Train four sets of training data

    std::cout << predict({0, 0})[0] << " " << predict({ 0, 0 })[1] << std::endl;//The predicted result is close to 1,1

    return 0;

}
```

By now, you may have realized that implementing neural networks in C++ can be quite challenging in terms of coding, but its computational speed is unmatched. This will become even more apparent as we progress. In C++, you need to carefully consider and design data structures, precision, and underlying resource allocation, which are aspects that languages like JavaScript and Python typically don't require or allow. However, these aspects are crucial for training deep neural networks. What struck me the most was when I was rewriting and testing a handwritten digit recognition model for this book. I initially wrote a JavaScript version for better understanding and tested it using some native JavaScript IDEs. A single gradient descent iteration with just ten training data points took at least 1-2 minutes. However, after rewriting the code in C++ and running it in Visual Studio, it could complete over 100 gradient descent iterations per second.

In conclusion, as I mentioned at the beginning of this book, JavaScript is closer to mathematical expressions and concise, making it easier to grasp simple concepts. Ultimately, to build truly large-scale models without excessive resource consumption, we inevitably need C++. In reality, C++ is not as difficult and complex as imagined. I often directly copy and paste entire JavaScript code blocks into the C++ compiler. Then, all I need to do is include the necessary header files, make some definitions, specify data types for each variable, replace JavaScript functions with their C++ counterparts, and finally fix any minor bugs reported by the compiler.

## Section 5: Common Misconceptions (Must Read)

In our learning journey so far, we've encountered a few misconceptions. While we've touched upon these points, I want to emphasize them specifically here. Sometimes, we get so focused on the formulas and code that we forget their origins, which hinders our deeper learning and understanding of neural networks and deep learning.

1. Neural networks themselves can be understood as functions where the output values depend on the input values. During forward propagation (feedforward), w and b are coefficients, while during backpropagation, w and b are independent variables. The prediction function of a neural network is not an implicit function or equation. A closed image on a plane is a projection of a 3D prediction function onto a plane (with a height of 0), often represented by different colors for different heights. The closed lines are the intersection lines between the 3D prediction function and the plane.
2. Neural networks that use the backpropagation algorithm can all be called BP neural networks, but not all neural networks use the backpropagation algorithm and gradient descent for training. Most neural networks belong to the category of feedforward neural networks. MLP, FCNN, BPNN, and various other NNs are not just parallel relationships. A neural network can have multiple names depending on its structure, the algorithms used, and other factors.
3. When training on an entire dataset (multiple data points), standard gradient descent performs backpropagation for each training data point. For each parameter, it records the gradient calculated during each data point's backpropagation and finally takes the average (Note: each data point's backpropagation is calculated separately, and the gradients are averaged. We do not average the parameters or input values to calculate all gradients!). Stochastic gradient descent, on the other hand, updates all parameters of the neural network using the gradients calculated from a single data point as soon as its propagation is complete. In practice, backpropagation often utilizes mini-batch stochastic gradient descent, which we'll discuss later. This method randomly selects a small batch of training data from the entire dataset, calculates the average gradient using standard gradient descent, and immediately updates the parameters. It then repeats this process with another small batch, and so on.
4. When calculating the gradients of all parameters in a neural network after a single data point's forward propagation, regardless of the number of parameters, only one backpropagation is needed to obtain the gradients of all parameters. If we don't use the backpropagation algorithm and instead calculate partial derivatives using small increments (i.e., adding a small increment h to the function's input value as we did when learning about derivatives, where h theoretically approaches zero but is usually a number like 0.000001 in code), we would need to perform forward propagation as many times as there are parameters, which is extremely resource-intensive. This is where the significance of backpropagation's chain rule for derivative calculation comes in.
5. The "standard gradient descent" mentioned above and below is also known as "Batch Gradient Descent (BGD)", where "Batch" refers to all data in the training set. Therefore, it's the same as our standard gradient descent. As for "Mini Batch," it simply means taking a small portion of the data from the entire dataset. It's crucial to remember what each term represents.
6. The order of backpropagation is irreversible! You cannot skip a layer and calculate the gradient of its previous layer. For example, consider a function f(x) = a(b(x)), where a and b are both functions. We cannot directly calculate the derivative of f without first finding the derivative of b.

Finally, I highly recommend that you revisit this entire chapter and skim through all the underlined and bold text. This will help you gain a holistic and comprehensive understanding of neural networks.


# Chapter 2: Improving Neural Networks

## Section 1: Better Algorithms

### 1.1 ReLU and Its Derivative Activation Functions

Sigmoid and Tanh activation functions share a common problem: vanishing gradients.

Recall the concept of gradients. When calculating the partial derivative of a neuron's activation with respect to its pre-activation, we differentiate the activation function. For instance, the derivative of sigmoid is sigmoid(x)\*(1-sigmoid(x)). Observe the graphs of sigmoid (green) and its derivative (blue):

![ ](./images/1698116595178.png)

Notice that sigmoid only exhibits significant output changes when the input is near 0. For inputs far from 0 on either side, the output approaches 0 or 1, respectively. Consequently, its derivative is significant only near 0, diminishing to almost zero further away. Moreover, the maximum derivative value at 0 is only 0.25. What does this imply?

If all neurons employ sigmoid as the activation function, numerous near-zero values will be multiplied during backpropagation. The resulting gradient will also be close to zero. As the neural network's depth increases, this issue exacerbates, potentially causing the compiler to treat the gradient as zero even with low precision. This is known as the **Vanishing Gradient** problem.

Tanh activation function offers a slight improvement but fundamentally suffers from the same issue of near-zero derivatives for inputs far from 0.

Enter the ReLU activation function. Recall its graph and formula. It outputs 0 for inputs less than 0 and the input itself for inputs greater than 0. Therefore, its derivative is 0 for inputs less than 0 and 1 for inputs greater than 0. This seems to circumvent the vanishing gradient problem as the derivative is consistently 1 for positive inputs. We can even bypass explicit calculations by simply comparing the input with 0 to determine its derivative (0 or 1), directly identifying neurons needing updates.

However, a new and glaring problem emerges. Directly outputting 0 for inputs less than 0 appears rather abrupt. In fact, a single abnormal input can render the neuron inactive. Let's investigate why:

Consider a neuron employing ReLU. Numerous normal inputs are processed, resulting in a positive sum (wx+b), which is directly outputted. Now, an abnormally large input, say x1=100, arrives, leading to a large sum, for instance, 200. However, if the target output is 1, the gradient generated on the corresponding weight w will be extremely large, say 100. After multiplying by the learning rate, w is updated to a very small value, such as -50. Subsequently, with normal inputs like 0.3 or 0.9, the sum is likely negative, causing ReLU to output 0 with a derivative of 0. Consequently, all parameters w and b of this neuron have zero gradients, preventing any further updates (unless another abnormal input occurs). The neuron effectively dies, essentially reverting to the vanishing gradient problem.

To mitigate this, we can employ a ReLU variant, the **Leaky ReLU**, as the activation function. The graph compares Leaky ReLU (red) with ReLU (blue):

![ ](./images/1698131465385.png)

Leaky ReLU is expressed as:

![ ](./images/1698131726284.png)

Here, α is a user-defined constant (e.g., 0.1, 0.05, 0.01), determining the slope of the function's negative input segment. As evident from the graph, Leaky ReLU avoids zero or near-zero derivatives. The derivative is 1 for positive inputs and α for negative inputs, effectively preventing neuron death. The concept is straightforward, so we'll omit the code example.

Subsequently, more effective activation functions emerged, many derived from ReLU. One such example is the **ELU (Exponential Linear Unit)**:

![ ](./images/1698132872218.png)

The orange curve represents ELU, while the purple curve depicts its derivative. Similar to the previous example, α (usually 1) controls the slope. ELU, like ReLU, avoids neuron death and vanishing gradients for normal (and all positive) input ranges. However, ELU exhibits greater smoothness compared to Leaky ReLU, which tends towards linearity when α approaches 1, posing challenges in complex training scenarios.  ELU inherits all of ReLU's advantages and, despite slightly higher computational cost for negative inputs, has become a widely adopted activation function.

Here's the ELU implementation in JavaScript:
```javascript
function elu(x) {
    if (x >= 0) {
        return x;
    } else {
        return 1.0 * (Math.exp(x) - 1);
    }
}

function elu_derivative(x) {
    if (x >= 0) {
        return 1;
    } else {
        return 1.0 * Math.exp(x);
    }
}
```

And in C++:
```cpp
auto elu = [](double x) {
    if (x >= 0) {
        return x;
    } else {
        return 1.0 * (exp(x) - 1);
    }
};

double elu_derivative(double x) {
    if (x >= 0) {
        return 1;
    } else {
        return 1.0 * exp(x);
    }
}
```

Finally, a point worth mentioning, particularly relevant to our current discussion. You've encountered "e" frequently throughout this book. While most readers might already know, "e" isn't merely an irrational number approximately equal to 2.718281828459. It's the base of the natural logarithm, implying that the derivative of e^x is e^x itself. The key takeaway here is that the frequent use of "e" in neural network algorithms primarily stems from simplifying derivative calculations during backpropagation.  Otherwise, would the derivatives of Sigmoid and ELU be mere coincidences?

We'll delve into more activation functions later.

### 1.2 Cross-Entropy Loss Function

You might have heard of "entropy." **Entropy** quantifies a system's disorder (originating from thermodynamics and later extending to broader mathematical domains). For instance, a deck of cards arranged in a specific order exhibits low disorder (low entropy), while shuffling them increases their disorder (high entropy). This is a qualitative description. To delve into quantitative entropy calculations, we need to understand "information content."

**Information content** intuitively refers to the value or significance of something to someone. For example, me telling you "Did you know 1+1=2?" right now probably holds little information content for you. However, explaining the "cross-entropy function" likely carries more information. We can make such judgments, but what's the basis?

Reflecting on these examples, you perceive low information content in "1+1=2" likely because you already know it. However, if you were (hypothetically) unaware or uncertain about it, my confirmation would carry more information. The key lies in your prior certainty, or the probability you assigned to its correctness before my statement. High certainty implies low information gain upon confirmation and vice versa.

Let's consider a more abstract numerical example. Event A has a 99% probability of occurrence, while event B has a 10% probability. If event A occurs, the information content is relatively low. Conversely, event B's occurrence carries more information.  Now, let's explore how to quantify information content.

Assume event C has a 99% probability of occurrence, implying a 1% chance of non-occurrence. Occurrence carries low information, while non-occurrence carries high information. Focusing solely on this event makes quantifying information content challenging. In reality, information content is a human-defined concept, not an inherent natural quantity. Therefore, instead of pondering its form, we should consider how to make it universally computable.

Firstly, consider event C. Let's introduce event D with a 50% probability for both occurrence and non-occurrence. The probability of their joint occurrence is the product of their individual probabilities, 99%\*50%, interpretable as conditional probability (though we won't express it that way for now).  Does their joint occurrence also possess information content?  Ideally, we desire the information content of their joint occurrence to equal the sum of their individual information contents. In other words, the information content of C and D occurring together equals the sum of the information content of C occurring and D occurring. The reason?  As mentioned earlier, it's a convenient definition for computational purposes.

However, a challenge arises. Joint occurrence is represented by the product of probabilities. How can we transform it into summation? This is where the logarithm (log) comes into play.  You might be familiar with its purpose, but let's briefly recap.

Logarithm is the inverse operation of exponentiation. loga(x) essentially asks "to what power must 'a' be raised to obtain 'x'?"  If a^x = N (where a>0 and a≠1), then x is the logarithm of N to the base 'a'. For instance, log2(8) seeks the power to which 2 must be raised to obtain 8, which is 3 (2^3 = 8).

Logarithms transform multiplication into addition: log(A\*B) = log(A) + log(B), irrespective of the base.  For example, log(4\*8) = log(4) + log(8). Using base 2, log(4) asks "2 to what power equals 4?", resulting in 2.  log(8) asks "2 to what power equals 8?", resulting in 3.  log(4\*8) is equivalent to log(32), asking "2 to what power equals 32?", resulting in 5.  We observe that 2 + 3 = 5, confirming log(4\*8) = log(4) + log(8).

The underlying principle is x^a \* x^b = x^(a+b), meaning a number raised to the power 'a' multiplied by the same number raised to the power 'b' equals the number raised to the power 'a+b'.  This should be familiar from elementary mathematics.

Now, to ensure the information content of events C and D occurring together equals the sum of their individual information contents, we need to incorporate logarithms in our calculation.  For instance, if event C has a probability of 0.99, its information content expression should include log(0.99). 

Two questions remain:

1. **What base should we use for the logarithm?**
2. **Should we multiply the logarithm by a coefficient or apply any further operations?**

For the first question, theoretically, any base is equivalent (not in terms of numerical results, but in achieving our objective). However, practically, as mentioned in the previous section, to facilitate derivative calculations during backpropagation, neural networks typically use the natural logarithm (ln), where the base is 'e'.  Interestingly, the derivative of ln(x) is 1/x, simplifying calculations.  You can find the proof online if you're interested.

For the second question, let's examine the graph of y = ln(x):

![ ](./images/1698479067117.png)

Notice that for inputs between 0 and 1 (the range of probabilities), the output y is always negative.  To obtain positive information content values, we multiply ln by -1: -ln(x).

Therefore, we arrive at the formula for information content:  If an event has a probability of 'p', its **information content = -ln(p)**.

![ ](./images/1698479719401.png)

Observing the graph of y = -ln(x), we see that it aligns with our initial intuition:  low probability events carry high information content upon occurrence, while high probability events carry low information content.

Let's return to entropy, which represents system disorder. High disorder implies the system consistently provides significant information, suggesting high entropy.  For instance, a game with a 50% chance of winning and losing exhibits high disorder. Conversely, a near-certain win (99.99%) signifies low disorder due to predictable outcomes.

How do we connect entropy and information content? Direct addition isn't suitable.  For example, an event with a 1% probability of occurrence carries high information content if it happens. However, if it doesn't occur, why include its information content in the entropy calculation? Therefore, we multiply the information content by the probability of occurrence.  Only actual occurrences contribute to the overall information provided by the system.

Thus, entropy is calculated as the sum of the product of each event's probability and its corresponding information content, plus the product of the probability of the event not happening and the information content of it not happening.  More formally: if an event has a probability of 0.9, implying a 0.1 probability of non-occurrence, its entropy is:

*0.9\*-ln(0.9) + 0.1\*-ln(0.1)*

Generalizing for a binary event system with probability 'p':

*-p\*ln(p) - (1-p)\*ln(1-p)*

Here, I've factored out the negative sign and used (1-p) to represent the probability of non-occurrence.

Plotting the graph of y = -x\*ln(x):

![ ](./images/1698482097001.png)

We see that entropy peaks at 0.5, aligning with our initial expectation.  When the probabilities of occurrence and non-occurrence are similar, the system exhibits high disorder, reflecting high uncertainty.

One final step remains! Having explored entropy and its calculation, we're ready for cross-entropy. This section might be conceptually challenging for beginners, so I've slowed down the pace, potentially repeating points for clarity. Ensure you grasp the preceding concepts before proceeding.

Recall the purpose of loss functions: comparing and quantifying the error between a neural network's predictions and the true values.  We calculated entropy to compare the error between two probability systems. One system is our neural network, while the other represents the true data distribution.  MSE (mean squared error) directly compares their predictions, deeming them equivalent when the squared difference is minimal.  Cross-entropy, however, compares their entropy values.

Let's first present the formula for cross-entropy in a single-class setting: Given a model's (neural network's) prediction 'x' and the true target value 't' (both probabilities between 0 and 1), **Cross Entropy (CE(Loss))** is:

*t\*l(x) + (1-t)\*l(1-x)*

where l() is the information content function: l(x) = -ln(x).

You'll notice the resemblance between cross-entropy and entropy, as expected.  The key difference lies in using the true value 't' instead of the predicted probability 'x' in the "information content times probability" term. However, the information content is still calculated using the predicted value 'x', hence the term "cross."  A more intuitive explanation for this "cross" aspect exists than using KL-divergence or relative entropy.

Typically, the true value 't' is either 0 or 1. Also, recall that a number multiplied by 0 equals 0, while multiplication by 1 yields the number itself. Examining the formula, l(x) represents the information content of the event occurring, while l(1-x) represents the information content of it not occurring.  But did the event actually occur or not?  That's where 't' comes in.  If t=0 (event didn't occur), we want to disregard the information content of it occurring, hence multiplying t\*l(x) results in 0. On the other hand, the right-hand term, representing the information content of non-occurrence, is what we want. Multiplying by (1-t) = 1 yields the information content of non-occurrence itself.  Based on the principles of information content, we know that larger (1-x) implies smaller information content. Therefore, smaller 'x' (closer to the true value 't') leads to smaller cross-entropy.  The reverse reasoning applies similarly.

In reality, 't' doesn't have to be strictly 0 or 1. It can take any value between 0 and 1. However, in such cases, the cross-entropy can never reach 0 (even when the prediction matches the true value), requiring MSE to check for convergence.  Cross-entropy then serves primarily for backpropagation.

This explanation provides an algebraic understanding of cross-entropy.  Online resources often introduce it through KL-divergence (relative entropy) if you wish to explore further.

Here's a Geogebra animation illustrating cross-entropy: https://www.geogebra.org/m/pp5eqsvy

In the animation, slider 'xi' represents the true value, 'yi' is the predicted value, the x-axis corresponds to the prediction, and the y-axis represents the cross-entropy loss.

Cross-entropy's derivative has several equivalent forms, sometimes appearing unfamiliar in different resources.  For instance, the derivative can be expressed as:

*(t-x) / (x^2-x)*

or

*-t/x + (1-t)/(1-x)*

Here's the JavaScript code for cross-entropy:
```javascript
// Cross-entropy function
function CE(out, out_hat){
    if(out_hat==out){ // Prevent log calculation errors
        return 0
    }else{
        return -(out*Math.log(out_hat)+(1-out)*Math.log(1-out_hat));
    }
}

// Cross-entropy derivative for backpropagation
function CED(out, out_hat){
    if(out_hat==out){
        return 0
    }else{
        return -out/out_hat + (1-out)/(1-out_hat)
    }
}
```

And in C++:
```cpp
// Cross-entropy function
double CE(double out, double out_hat){
    if(out_hat == out){ // Prevent log calculation errors
        return 0;
    }else{
        return -(out * log(out_hat) + (1 - out) * log(1 - out_hat));
    }
}

// Cross-entropy derivative for backpropagation
double CED(double out, double out_hat){
    if(out_hat == out){
        return 0;
    }else{
        return -out / out_hat + (1 - out) / (1 - out_hat);
    }
}
```

Why use cross-entropy?  What advantages does it offer over MSE or MLE?  This Geogebra animation demonstrates: https://www.geogebra.org/m/nx9wsb32

Setting the true target value to 0.25, the x-axis represents the prediction, and the y-axis represents the loss. The three curves correspond to MSE, MLE (maximum likelihood estimation), and CELoss (cross-entropy) respectively.

![ ](./images/1698499473936.png)

Green represents cross-entropy, orange represents MSE (both seeking minima), and blue represents MLE (seeking maxima). The blue dot's x-coordinate represents the prediction.  When it reaches the minimum (or maximum for MLE) loss, its x-coordinate aligns with the true value 'yi'.  Observe the black tangent lines and their slopes (k) at these points.  The slope represents the partial derivative of the loss with respect to the prediction during backpropagation, a crucial component of gradient calculations. We desire larger gradients for faster convergence but not so large as to cause explosions, rendering the model useless.  Experiment by dragging the slider.  You'll notice that regardless of the true and predicted values, the slope of cross-entropy always exceeds that of MLE (multiplied by -1), which in turn exceeds the slope of MSE.  In other words, ▽CELoss > ▽MLE > ▽MSE, where ▽ denotes the gradient symbol.  Therefore, cross-entropy facilitates faster and more effective model convergence.

This section has been quite extensive, likely the most challenging since gradient descent and backpropagation.  However, cross-entropy's significance in enhancing model training makes it a crucial concept to grasp.

### 1.3 Milestone: Training a Single-Class Handwritten Digit Recognition Model

With the knowledge gained, we can attempt a practical implementation: a small-scale handwritten digit recognition model to determine if a handwritten letter is "X." We'll use C++ for this example due to its superior performance compared to JavaScript.  As mentioned earlier, if you're unfamiliar with C++, it's time to gradually incorporate it into your learning.  C++ can be hundreds of times faster than JavaScript and will be our primary language for practical implementations later in the book.  If you use a mobile device, you can use the free c4droid to run c\+\+code, which also supports a relatively complete range of functions; On the computer side, you can usually use free software such as Visual Studio 2022.

```cpp
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>
#include <sstream>

using namespace std;

//Save the neural network weight bias parameters to a binary file
void saveNetwork(const std::vector<std::vector<std::vector<std::vector<double>>>>& tensor, const std::string& filename) {
	std::ofstream out(filename, std::ios::binary);
	for (const auto& three_dim : tensor) {
		for (const auto& two_dim : three_dim) {
			for (const auto& one_dim : two_dim) {
				for (const double& value : one_dim) {
					out.write(reinterpret_cast<const char*>(&value), sizeof(double));
				}
			}
		}
	}
	out.close();
}

void loadNetwork(const std::string& filename, std::vector<std::vector<std::vector<std::vector<double>>>>& tensor) {
	std::ifstream in(filename, std::ios::binary);
	for (auto& three_dim : tensor) {
		for (auto& two_dim : three_dim) {
			for (auto& one_dim : two_dim) {
				for (double& value : one_dim) {
					in.read(reinterpret_cast<char*>(&value), sizeof(double));
				}
			}
		}
	}
	in.close();
}

//Global variables
std::vector<std::vector<std::vector<std::vector<double>>>> network;//Due to the fact that C++vectors do not allow different types of data (i.e. vectors and doubles in C++) to be nested in each layer like arrays in JS, it is necessary to add a {} to the bias coefficient to make it also a vector. For details, please refer to the network assignment in the main function
std::vector<std::vector<double>> networkn;
std::vector<std::vector<double>> networkb;
double rate;//Learning rate
double aim;//Target loss value

//Quickly generate vectors of specified length with all values equal to 0.1
std::vector<double> generateVector(int length) {
    std::vector<double> result(length, 0.1);
    return result;
}

//A neuron
vector<double> neuron(std::vector<double> w, std::vector<double> x, double b) {
    //Using Elu as the activation function
    auto elu = [](double x) {
        if (x >= 0) {
            return x;
        } else {
            return 1.0 * (exp(x) - 1);
        }
    };

    //Sum ∑ wx+b
    auto sigma = [&w, &x, b]() {
        double sum = 0;
        for (int i = 0; i < w.size(); i++) {
            sum += w[i] * x[i];
        }
        return sum + b;
    };
	
    double sum = sigma();//Value before activation
    return { sum,elu(sum) };

}

vector<double> S_neuron(std::vector<double> w, std::vector<double> x, double b) {
    //Use sigmoid as the activation function (will be learned later)
    auto sigmoid = [](double x) {
        return 1 / (1 + exp(-x));
    };
	
    //Sum ∑ wx+b
    auto sigma = [&w, &x, b]() {
        double sum = 0;
        for (int i = 0; i < w.size(); i++) {
            sum += w[i] * x[i];
        }
        return sum + b;
    };

    double sum = sigma();//Value before activation
    return { sum,sigmoid(sum) };

}

double sigmoid_derivative(double y) {
    return y * (1 - y);
}

double elu_derivative(double x) {
    if (x >= 0) {
        return 1;
    } else {
        return 1.0 * exp(x);
    }
}

double CE(double out, double out_hat) {
    if (out_hat == out) {
        return 0;
    }else {
        return -(out * log(out_hat) + (1 - out) * log(1 - out_hat));
    }
}

double CED(double out, double out_hat) {
    if (out_hat == out) {
        return 0;
    } else {
        return -out / out_hat + (1 - out) / (1 - out_hat);
    }
}

//Prediction - Forward propagation
vector<double> predict(vector<double> content) {
    //Spread forward to the hidden layer
    for (int m = 0; m <= networkn[0].size() - 1; m++) {
        auto r0 = neuron(network[0][m][0], content, network[0][m][1][0]);//Unlike JS, [0] needs to be added at the end for the reason stated in the global variable declaration
        networkb[0][m] = r0[0];//Inactive value
        networkn[0][m] = r0[1];//Activation value
    }

    //Spread forward to the output layer
    for (int n = 0; n <= networkn[1].size() - 1; n++) {
        auto r1 = S_neuron(network[1][n][0], networkn[0], network[1][n][1][0]);
        networkb[1][n] = r1[0];//Inactive value
        networkn[1][n] = r1[1];//Activation value
    }
    return networkn[1];
}

//Training backpropagation stochastic gradient descent
double trainNet(vector<vector<double>> dt) {
    std::vector<double> out_hat = predict(dt[0]);//Prediction
    double CEError = 0;
    for (int l = 0; l <= out_hat.size() - 1; l++) {
        CEError += CE(dt[1][l], out_hat[l]);
    }
    CEError = CEError / dt.size();
    
    std::vector<double> rMEdN;
    for (int l = 0; l <= out_hat.size() - 1; l++) {
        rMEdN.push_back(rate * CED(dt[1][l], out_hat[l]) * sigmoid_derivative(networkn[1][l]));
    }

    double sum = 0;
    for (int i = 0; i < rMEdN.size(); i++) {
        sum += rMEdN[i];
    }
    double rMEdNA = 0;
    if (rMEdN.size() > 0) {
        rMEdNA = sum / rMEdN.size();
    }

    //Update output layer weights
    for (int p = 0; p <= networkn[1].size() - 1; p++) {
        for (int q = 0; q <= network[1][p][0].size() - 1; q++) {
            network[1][p][0][q] -= rMEdN[p] * networkn[0][q];
        }
    }

    //Update output layer bias
    for (int p = 0; p <= networkn[1].size() - 1; p++) {
        network[1][p][1][0] -= rMEdN[p];
    }

    //Update hidden layer weights
    for (int p = 0; p <= networkn[0].size() - 1; p++) {
        for (int q = 0; q <= network[0][p][0].size() - 1; q++) {
            double averagenN = 0;
            for (int s = 0; s <= network[1].size() - 1; s++) {
                averagenN += network[1][s][0][p];
            }
            averagenN = averagenN / network[1].size();
            network[0][p][0][q] -= rMEdNA * averagenN * elu_derivative(networkb[0][q]) * dt[0][q];
        }
    }

    //Update hidden layer bias
    for (int p = 0; p <= networkn[0].size() - 1; p++) {
        double averagenN = 0;
        for (int s = 0; s <= network[1].size() - 1; s++) {
            averagenN += network[1][s][0][p];
        }
        averagenN = averagenN / network[1].size();
        network[0][p][1][0] -= rMEdNA * averagenN * elu_derivative(networkb[0][p]);
    }
    return CEError;
}

void train(vector<vector<vector<double>>> dt) {
    int i = 0;
    while (true) {
        i++;
        double err = 0;
        for (int c = 0; c <= dt.size() - 1; c++) {
            double preErr = trainNet(dt[c]);
            err += preErr;
        }
        if (i % 1000 == 0) {
            rate *= 10;//Due to the fact that this model approaches the optimal gradient in the later stages of training, the learning rate is increased, and more effective adaptive learning rate algorithms will be learned later
        }
        if (err <= aim) {
            std::cout << "Training completed with err <= " << aim << " (" << err << ")" << std::endl;
            std::cout << ">>> finished " << dt.size() * i << " steps (" << i << " rounds) gradient descent in " << /*elapsed + */"ms <<<" << std::endl;
            break;
        } else {
            std::cout << "Round: " << i << " Training: " << dt.size() * i << " CEloss: " << err << std::endl;
        }
    }
}

int main() {
    network = {
    {
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}},
        {{generateVector(49)},{0.1}}
    },
    {
        {{generateVector(25)},{0.1}}
    }
    };
	
    networkn = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0}
    };
    networkb = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0}
    };

    rate = 0.03;//Learning rate
    aim = 1e-10;//Target loss value

    train({
    {{
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
    },{0}},
    {{
        1,0,0,0,0,0,1,
        0,1,0,0,0,1,0,
        0,0,1,0,1,0,0,
        0,0,0,1,0,0,0,
        0,0,1,0,1,0,0,
        0,1,0,0,0,1,0,
        1,0,0,0,0,0,1,
    },{1}},
    {{
        1,1,1,1,1,1,1,
        1,0,0,0,0,0,1,
        1,0,0,0,0,0,1,
        1,0,0,0,0,0,1,
        1,0,0,0,0,0,1,
        1,0,0,0,0,0,1,
        1,1,1,1,1,1,1,
    },{0}},
    {{
        0,1,0,0,0,0,0,
        0,1,1,0,0,1,0,
        0,0,1,0,1,0,0,
        0,0,1,1,1,0,0,
        0,0,1,0,1,0,0,
        0,0,0,0,0,1,1,
        0,1,0,0,0,0,0,
    },{1}},
    {{
        1,1,0,0,0,0,0,
        0,1,1,0,0,1,0,
        0,0,1,0,1,0,0,
        0,0,0,1,0,0,0,
        0,0,1,0,1,0,0,
        0,1,0,0,0,1,0,
        0,1,0,0,0,0,1,
    },{1}},
    {{
        1,1,0,0,0,0,0,
        0,1,0,0,0,1,1,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,1,1,
        0,1,0,0,0,0,0,
    },{0}},
    {{
        0,0,0,0,0,0,0,
        0,1,0,0,0,1,1,
        1,0,0,0,0,0,1,
        1,1,1,1,1,1,1,
        1,0,0,0,0,0,1,
        0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,
    },{0}},
    {{
        0,0,0,0,0,0,0,
        0,1,0,0,1,0,0,
        0,0,1,1,0,0,0,
        0,0,1,0,1,0,0,
        0,0,1,0,0,1,0,
        0,1,0,0,0,0,0,
        0,1,0,0,0,0,0,
    },{1}},
    {{
        1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,
        1,1,1,0,1,1,1,
        1,1,0,0,0,1,1,
        1,1,1,0,1,1,1,
        1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,
    },{0}},
    {{
        0,0,0,0,0,0,0,
        1,0,0,0,1,0,0,
        0,1,0,1,0,0,0,
        0,0,1,0,0,0,0,
        0,1,0,1,0,0,0,
        1,0,0,0,1,0,0,
        0,0,0,0,0,0,0,
    },{1}},
    {{
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,
        0,0,0,1,0,1,0,
        0,0,0,0,1,0,0,
        0,0,0,1,0,1,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
    },{1}},
    {{
        0,0,0,0,0,1,0,
        0,0,1,0,1,0,0,
        0,0,0,1,0,0,0,
        0,0,1,0,1,0,0,
        0,0,0,0,0,1,0,
        0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,
    },{1}},
    {{
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,
        0,1,0,0,1,0,0,
        0,0,1,1,0,0,0,
        0,0,1,0,1,0,0,
        1,1,0,0,0,1,0,
    },{1}},
    {{
        0,1,0,1,0,0,1,
        0,0,0,1,0,0,0,
        1,0,1,0,0,0,0,
        0,1,0,0,1,0,0,
        0,0,0,1,1,0,1,
        0,0,1,0,1,0,0,
        1,0,0,1,0,1,0,
    },{0}}
    });

    saveNetwork(network, "./model_1.bin");//Save the weights and bias parameters of your trained model to a binary. bin file, where the path is changed to your own storage path
    /*After completing the training, you can delete the train() above and use it instead for the next use
    LoadNetwork("model_1.bin", network);//Change the path here to your own storage path
    Read model weights and bias parameters directly from the saved. bin binary file, and then use
    double result = predict({
    	0,0,0,0,0,0,0,
        0,1,0,0,0,1,0,
        0,0,1,0,1,0,0,
        0,0,0,1,0,0,0,
        0,0,1,0,1,0,0,
        0,1,0,0,0,1,0,
        0,0,0,0,0,0,0,
    })[0];
    std::cout << result << std::endl;
    Change 0 to 1 to represent white, and the model will predict whether the letter X is included
    */

    return 0;
}
```

You can enhance accuracy by adding more training data or adapt this framework for other tasks like digit recognition, playing number games to identify patterns, etc.


## Section 2: Entering the World of Multi-Class Classification

### 2.1 Introduction

Let's briefly discuss multi-class classification. Previously, we primarily focused on neural networks with a single output neuron, like this:

![ ](./images/1698827561583.png)

Such networks perform single-class classification, such as predicting the probability of a handwritten digit image being "1" or detecting the presence of a tree in an image. Today, we'll enable multi-class classification, predicting probabilities for each digit (0-9) in a handwritten image or identifying the probabilities of an image depicting a tree, flower, or grass. This requires multiple output neurons:

![ ](./images/1698836962519.png)

Each output neuron in a multi-output network is fully connected to every neuron in the preceding hidden layer. During backpropagation, a hidden layer neuron connects to multiple output neurons.  Consequently, each connected output neuron influences how to adjust this hidden neuron's activation (essentially modifying its 'w' and 'b').  

Referring to the diagram, to optimize n6 and n7, backpropagation involves adjusting weights between hidden and output layers, which doesn't cause conflicts. However, errors also propagate to hidden layer neurons, further distributed to weights between input and hidden layers (omitting bias for now). Consider neuron n3, connected to n6 and n7. Changes in its output affect the error loss from both n6 and n7.  Hence, we have two ▽n3: ∂n6/∂n3 and ∂n7/∂n3. Typically, we sum (or average) these to obtain the final ▽n3, which is ∂loss/∂n3. Note that n3 refers to its activation value.  After obtaining the combined gradient, we can proceed with backpropagation to modify 'w' and 'b' as before.

Exercise extreme caution here, especially during code implementation, as chain rule errors are common.  I struggled with this concept for a while when learning neural networks.  Let's reiterate:  

Taking the previous example, ∂n6/∂n3 should be expanded as ∂n6/∂z6 * ∂z6/∂n3, where n6 is the activation of neuron 6, and z6 is its pre-activation value (wx+b).  Based on our previous knowledge, ∂n6/∂z6 involves differentiating the activation function, while ∂z6/∂n3 equals the weight w36 connecting neurons 3 and 6.  The crucial point is that summing the backpropagated gradients from two output neurons, ∂n6/∂n3 and ∂n7/∂n3, and expanding them (assuming activation function derivatives ∂n6/∂z6 and ∂n7/∂z7 are k6 and k7, respectively) should yield:

*k6\*w36 + k7\*w37*

**Do not** mistakenly write it as the sum of k6 + k7 separately multiplied by w36 and w37.  Firstly, this is mathematically incorrect.  Secondly, after introducing softmax normalization, k6 + k7 often equals 0, resulting in zero gradients and preventing updates.  Be mindful of this potential pitfall.

### 2.2 Softmax Normalization

Before delving into Softmax, let's revisit probability.  Suppose we task a neural network with identifying the presence of a tree in an image. It essentially outputs a probability between 0 and 1 (achieved using the sigmoid activation function on the output neuron). Probabilities closer to 0 indicate a lower likelihood of a tree, while values closer to 1 suggest a higher likelihood.

Now, imagine we want to determine if the image depicts a tree, flower, or grass.  A predictive neural network would require three output neurons, each representing the probability of one of these classes.  However, what activation function should these neurons employ?  While sigmoid can scale each output to 0-1, consider outputs like 0.6, 0.5, and 0.3.  Their sum (1.4) doesn't equal 1, violating the probability constraint.  We need the sum of activations from these three neurons to equal 1, representing probabilities.  This is where normalization comes in.

You might have already envisioned the solution. If not, pause and consider the problem's essence. Can you anticipate how we might normalize the results?

We simply divide each activation value by the sum of all activation values. This represents the proportion each activation contributes to the total.  For instance, given activations 0.6, 0.5, and 0.3, their sum is 1.4.  Dividing each by 1.4 yields approximately 0.42, 0.35, and 0.21, achieving normalization (with minor rounding errors).

Next, we introduce the exponential function with base 'e'.  Firstly, it facilitates derivative calculations during backpropagation. Secondly, it amplifies larger values and accommodates negative inputs (as e^x is always positive).  This leads us to the **Softmax (Normalized Exponential Function)**:

![ ](./images/1700829181619.png)

To elaborate (exp(x) denotes e^x), the numerator 'xi' represents the pre-activation value (wx+b) of the output neuron being activated. The denominator sums the exponential of all output neurons' pre-activation values, including the current neuron's exp(xi).  

For example, given three pre-activation values z1, z2, and z3, the first neuron's activation is:

*exp(z1) / (exp(z1) + exp(z2) + exp(z3))*

The second and third neurons follow the same formula, with only the numerator's 'z' value changing to z2 or z3 while the denominator remains constant.

Now, let's visualize the Softmax function's graph.  Considering three pre-activation values (x, z2, and z3), we'll focus on 'x' as the variable being activated. The relationship between 'x' and its activation represents the Softmax function:

![ ](./images/1700830419486.jpg)

You might notice a resemblance to the sigmoid function.  Let's consider an alternative representation of sigmoid:

![ ](./images/1700830862447.png)

Our previous sigmoid formula is the first one. Multiplying both numerator and denominator by e^x:

![ ](./images/1700832289422.png)

We arrive at the second form.  The '1' in the denominator corresponds to the (e^z2 + e^z3) term in the Softmax function's denominator.

Here's a Geogebra animation of the Softmax function: https://www.geogebra.org/m/gdhw6n9k

Below is the C++ implementation of Softmax:
```cpp
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

// Calculate each activation value (numerator divided by denominator)
vector<double> CalculateActivation(std::vector<double> output, double sum){
	std::vector<double> yhat(output.size(), 0);
	for(int i=0; i<=output.size(); i++){
		yhat[i] += exp(output[i]) / sum;
	}
	return yhat;
}

vector<double> softmax(std::vector<double> input){
    // Calculate denominator
    double sum = 0;
    for(int i=0; i<=input[1].size(); i++){
    	sum += exp(input[1][i]);
    }
    vector<double> out = CalculateActivation(input[1], sum); // Calculate each activation
	return out;
}
```


### 2.3 Combining and Differentiating Softmax and Multi-Class Cross-Entropy

When training multi-class neural networks using backpropagation, we typically employ Softmax as the activation function in the output layer and multi-class cross-entropy as the loss function. This is because cross-entropy not only provides larger gradients but also simplifies the combined derivative when used with Softmax.

Let's first examine multi-class cross-entropy.  Recall that the single-class cross-entropy formula is *loss = t\*l(x) + (1-t)\*l(1-x)*, where 'x' is the prediction, 't' is the true value, and l() calculates information content: l(x) = -ln(x).

In multi-class settings, each output neuron contributes to the error loss. However, we can't directly apply the above formula.  Softmax ensures that the sum of all output neurons' predictions equals 1. The previous formula, with terms like (1-t) and (1-x), assumes each neuron's output falls between 0 and 1 individually.  The solution is simple: we discard the (1-t)\*l(1-x) term, resulting in *loss = t\*l(x)* for a single output neuron. Expanding l(x) gives us:

*loss = -t\*ln(x)*

Two potential issues might arise here, which I grappled with initially:

**Issue 1:**  When the true value 't' is 0, wouldn't any prediction 'x' lead to a loss of 0? (as shown below)

![ ](./images/1702108849083.png)

**Explanation:**  Indeed, the loss would be 0.  However, remember that we're using Softmax.  To increase one neuron's prediction while decreasing others, we have two options:

1.  Directly reduce the pre-activation values of other neurons, making the desired prediction stand out.
2.  Increase the desired neuron's pre-activation value, overshadowing the relatively smaller predictions of others.

For example, suppose three output neurons have pre-activation values (z1, z2, z3) of 1, 1, and 1.  After Softmax activation, their predictions become 0.33, 0.33, and 0.33. To boost the first prediction, we could either drastically reduce z2 and z3, yielding results like 0.98, 0.01, and 0.01 after activation. Alternatively, significantly increasing z1 would achieve a similar outcome.  While we typically employ both methods for optimization, multi-class cross-entropy differs.  We primarily rely on the second method, increasing the desired neuron's pre-activation value.  We don't directly minimize z2 and z3 (their losses are already 0 when their true values are 0).  Instead, we focus on maximizing z1 (its loss is non-zero when its true value is 1, though very small due to the minuscule contributions from z2 and z3).  This approach is effective overall.  Furthermore, when we analyze the combined derivative of Softmax and multi-class cross-entropy, we'll see that all neurons still get optimized.

**Issue 2:**  When the true value 't' and prediction 'x' are both 1, the loss is 0. However, the gradient isn't 0. Wouldn't this optimize the prediction 'x' towards negative values? (as shown below)

![ ](./images/1702110371360.png)

**Explanation:**  Remember that cross-entropy incorporates the exponential function e^x.  Regardless of whether 'x' is positive or negative, e^x is always greater than 0.  Therefore, we won't optimize the prediction towards negative values but rather optimize the pre-activation value 'z' towards negative infinity, which is reasonable.  (Here's a reminder of the y = e^x graph if needed)

![ ](./images/1702110642472.png)

With those issues addressed, let's combine Softmax and multi-class cross-entropy.

Consider the output layer of this neural network.  Let's analyze the forward pass:

![ ](./images/1702540024861.png)

Before activation, the three output neurons calculate ∑wx+b values of z1, z2, and z3, respectively. Applying Softmax activation:

*a1 = e^z1 / (e^z1 + e^z2 + e^z3)
a2 = e^z2 / (e^z1 + e^z2 + e^z3)
a3 = e^z3 / (e^z1 + e^z2 + e^z3)*

Remember that, for instance, z1 affects not only the numerator of a1 but also the denominators of all three activations.

Next, these activations (the network's predictions) are compared with the true values t1, t2, and t3 using multi-class cross-entropy:

*e1 = -t1\*ln(a1)
e2 = -t2\*ln(a2)
e3 = -t3\*ln(a3)*

Finally, we sum these individual losses to obtain the network's total error:

*cost = e1 + e2 + e3*

That concludes the forward pass, yielding the error.  Now, we need to backpropagate this error to calculate gradients for z1, z2, and z3, which will then be used to update weights and biases.

The backpropagation process involves mathematical derivations. Let's focus on z1 as an example.  We need to determine its gradient, ∂cost/∂z1. Remember that z1 influences e1, e2, and e3 as it appears in their Softmax denominators.  Therefore, we first calculate ∂cost/∂a1, ∂cost/∂a2, and ∂cost/∂a3.

*▽a1
 = ∂cost/∂a1
 = ∂cost/∂e1 \* ∂e1/∂a1
 = 1 \* -t1/a1
 = -t1/a1*

where -t1/a1 is derived from the derivative of -ln(x), which is -1/x. The calculations for ∂cost/∂a2 and ∂cost/∂a3 are analogous.

Next, we need the partial derivatives of a1, a2, and a3 with respect to z1.  This requires separate cases since z1 appears in both the numerator and denominator of a1's Softmax formula, while it's only in the denominators of a2 and a3.

**Case 1:**

*∂a1/∂z1
 = Softmax(z1)\*(1-Softmax(z1))
 = a1\*(1-a1)*

This derivation is similar to that of Sigmoid and can be found in detail online.

**Case 2:**

*∂a2/∂z1
 = a1\*a2*

*∂a3/∂z1
 = a1\*a3*

Combining all the derivatives:

*▽z1
 = ∂cost/∂z1
 = ∂cost/∂a1 \* ∂a1/∂z1 + ∂cost/∂a2 \* ∂a2/∂z1 + ∂cost/∂a3 \* ∂a3/z1
 = (-t1/a1) \* a1\*(1-a1) + (-t2/a2) \* a1\*a2 + (-t3/a3) \* a1\*a3
 = -t1\*(1-a1) + t2\*a1 + t3\*a3
 = -t1 + t1\*a1 + t2\*a1 + t3\*a3
 = -t1 + a1\*(t1+t2+t3)*

Since Softmax ensures normalized predictions and true values, only one of t1, t2, and t3 equals 1, while the rest are 0.  Therefore, t1 + t2 + t3 = 1.

*▽z1
 = -t1 + a1\*(t1+t2+t3)
 = -t1 + a1\*1
 = a1 - t1*

That's the calculation for ▽z1.  Similarly:

*▽z2 = a2 - t2
▽z3 = a3 - t3*

With these gradients, we can continue backpropagation to update weights and biases. For example, if a hidden neuron with activation 'x' connects to output neuron 1 with weight 'w', then ▽w = ▽z1\*x, and so on.

Thus, the seemingly complex derivative of the Softmax and multi-class cross-entropy combination simplifies to an elegant a-t, significantly reducing computational cost.

### 2.4 Milestone: Training a Multi-Class Digit Recognition Model

We've enhanced our neural network and incorporated multi-class classification. Now, we can train a model to recognize handwritten digits from 0 to 9.

Due to code length, we'll omit it here.  Please refer to the [book's GitHub repository (https://github.com/VeritNet/AI-Learning)](https://github.com/VeritNet/AI-Learning) and locate ./src/num_predict.cpp. Download it and run it using C++.  You'll also need the MNIST dataset, a free resource containing numerous 28\*28 pixel grayscale images of handwritten digits. It includes 50,000 training images and 10,000 test images to evaluate our trained model.

For convenience, I've extracted 1,000 training images and 100 test images into a text file located in the ./src/data folder of the GitHub repository.  Download the 'data' folder and place it in the same directory as num_predict.cpp.  The C++ code includes functions to read and parse this data into vectors for neural network input.  You can also download the original dataset from the [official MNIST website (http://yann.lecun.com/exdb/mnist/)](http://yann.lecun.com/exdb/mnist/) and write your own code to convert the images.  Simply store all grayscale pixel values sequentially in a vector.

The provided code is not yet optimized and might run slowly. Adjust the learning rate (the 'rate' variable in the 'main' function) as needed.  During training, the network might encounter plateaus.  Initially, set the target loss ('aim' variable in the 'main' function) to around 100 and use a small learning rate, such as 0.0003.

After training, the `saveNetwork(network, "./num_predict.bin");` function can save the trained model's weights and biases (as vectors) to a binary file. To resume training from a saved model, use `loadNetwork("./num_predict.bin", network);` at the beginning of the 'main' function.  This reads and assigns the model data to the 'network' variable.  You can then gradually decrease the target loss ('aim') and increase the learning rate ('rate') until the desired loss is achieved (e.g., 0.001).  However, avoid extremely low losses as they might indicate overfitting, a concept we'll explore later.

I've also included a simple HTML handwriting tool at ./src/num.html.  Draw a digit, and its grayscale representation appears in the input field below.  Copy this data, create a new text file alongside num_predict.cpp, paste the data, and update the file path in the prediction line within num_predict.cpp's 'main' function to use your file. Running the program (with a trained model) will predict the digit you drew (outputting a vector to the console, with each element representing the probability of the corresponding digit).

My trained model is available at ./src/num_predict.bin in the repository for your testing purposes.


## Section 3: Code Optimization: Multithreading and CPU Instruction Sets

To maximize performance, we can leverage CPU instruction sets like SSE2, AVX2, or AVX512 on Intel CPUs and employ multithreading during training and inference.

We'll utilize data parallelism and **Mini-batch Stochastic Gradient Descent (MSGD)**, an extension of SGD.  Instead of processing one data point at a time, we'll backpropagate through multiple data points simultaneously.  For each network parameter, we'll sum (or average) the gradients calculated from each data point and use the combined gradient for updates.

We'll create a thread pool, assigning each thread the task of backpropagating through a subset of the data. Upon completion, threads return their calculated gradients to the main thread, which waits for all threads before updating model parameters.

For demonstration, we'll use the SSE2 instruction set.  The code is located at ./src/num_predict_fast.cpp in the repository.  You'll also need ./src/fast.h, which contains SSE2-based functions used by num_predict_fast.cpp.

# Conclusion

In this first book, you've gained a foundational understanding of neural networks, delved into their design principles, explored improvements, and engaged in practical implementations.  In the next book, we'll venture into deep learning, exploring more powerful models built upon neural networks, and progressively bridge the gap between mathematical theory and real-world engineering applications to address your specific needs or explore new frontiers.

This book is open-source and freely available on GitHub.  If you have any questions, feel free to contact the author via email, YouTube.  You can also engage in discussions within the repository or other relevant platforms.

You're welcome to cite this book for academic or other purposes through its GitHub URL.  For non-commercial use and adherence to the CC BY-NC-SA 4.0 license, you're free to reproduce parts of the content without explicit permission.  I hope this book has been helpful in your learning journey.

Acknowledgments to all authors who generously share their knowledge in machine learning online and through other avenues.

--VeritNet AI Learning