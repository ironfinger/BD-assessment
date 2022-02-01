ah fair
idk kek
looks p similar to mine
wait
nah
ure not aggregating it
normal.select().agg(*[_var(col(c)).alias(c) for c in Normal.columns]) 
maybe
Tom H — Today at 6:21 PM
Perhaps
I have one Idea to fix it
so I'll try that
then do ur idea
Shonty — Today at 6:27 PM
i mean it isnt an idea kek
u need aggregate
cause variance is a aggregate func


# What is an RDD

- Resilient Distributed Dataset
- These are the elements that run and operate on multiple nodes to do parallel processing on a cluster. 
- These are immutable elements which means once you create one, you cannot change it. 
- You can apply multiple operations on these RDDs to achieve a certain task:
    - To apply operations on these RDD's there are two ways:
        - Transformation and Action

- Transformation: Are applied to create a new one (filter groupBY and map)
- Action: These are the operations that are applied, instructs spark to perform computation and send the result back to the driver.



# Key value pair
- Spark provieds special types of operations on RDD

# RDD Transformations: 
- Resilient Distributed Dataset -> one of first fundamental data structures 
