## What we did

We studied and attempted to implement machine learning and reinforcement algorithms that are potentially helpful to our score-following system using PyTorch.

## Who did what

### Marshall

Most of my time was spent studying reinforcement learning techniques. More specifically, policy gradient methods, which is a very hot area of research currently, and is the class of algorithm that a recent paper on Score Following uses. This involved watching the lectures for (Stanford's CS234 course)[http://web.stanford.edu/class/cs234/CS234Win2019/index.html] lectures 8 and 9 on Policy search. This was then followed by reading Chapter 13 from (Sutton and Barto's 2018 Reinforcement Learning Textbook)[http://incompleteideas.net/book/RLbook2018.pdf]. Finally, I followed along a [tutorial](https://towardsdatascience.com/breaking-down-richard-suttons-policy-gradient-9768602cb63b) on how to implement one of these algorithms on OpenAI's LunarLander task.

### Ryan

I had no background with PyTorch, so I decided to complete beginner tutorials on the technology. In the blitz tutorial (https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html), I learned the basics of navigating pytorch. In the custom data tutorial(https://pytorch.org/tutorials/beginner/data_loading_tutorial.html), I worked with a dataset that contained pictures of faces and implemented a model that mapped facial landmarks. In the reinforcement learning tutorial(https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), I learned more about neural networks in the context of solving the CartPole problem, where a cart must decide to move left or right in order to balance a pole that is on top of it.


## Learnings

* PyTorch basics and applications for data loading and reinforcement learning
* Reinforcement Learning - Policy Gradient methods (REINFORCE with Baseline)

## Inspiration/Relation to project

The reinforcement learning aspect of this hack-a-thing is relevant to our project because we believe that, and recent research suggests that, reinforcement learning might be a better solution than the existing probablistic models that exist (and actually perform decently).


## What didn't work

Initially, we wanted to tackle the problem of score following, but due to the short time constraint, we realized that there was no possible way that we could solve this problem for which both of us have very little experience in. Instead, we decided to dial it back and do more research/studying on potential methods and algorithms.
