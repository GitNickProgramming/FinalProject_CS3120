## <center> CS3120 - Machine Learning Final Project</center>
### <center>Traveling Salesman Problem (TSP) - Genetic Algorithm</center>
##### <center> Nick Gagliardi </center>

---
### To run the program:
```python
cd src
python main.py -v 1 --pop_size 500 --tourn_size 50 --mut_rate 0.02 --n_gen 20 --cities_fn '../data/cities.csv'
```

---
#### I. Description of the Problem
1. *What is the project about?*

    This project started as an idea to play around with reinforcment learning as well as the Traveling Salesman Problem (TSP). Find the most efficient route for my fiance and I to take while we travel around the United States. 
    
    Utilizing a Genetic Algorithm the program will attempted to find the shortest route between 15 American locations:
    - Denver, CO
    - Colorado Springs, CO
    - Telluride, CO
    - Las Vegas, NV
    - Grand Canyon, AZ
    - Yellowstone National Park, WY
    - Mount Rushmore, SD
    - Seattle, WA
    - Redwood National Park, CA
    - San Diego, CA
    - Los Angeles, CA
    - Mount Hood National Forest, OR
    - Santa Fe, NM
    - Chicago, IL
    - New York, NY

---
#### II. Implementation
1. **`./src/main.py`**
    - Coming Soon
2. **`./src/tsp_ga.py`**
    - Selection Technique - Tournament Selection
        - Initialized the population with the tournament size = 50
        - Save every individual for the tournament to find the fittest
        - Find the fittest individual according to the the distance between the cities in kilometers (because miles make no sense anywhere else in the world). 
3. **`./src/utils.py`**
    - Coming Soon
4. **`./data/cities.csv`**
    - Coming Soon
---
#### III. Libraries/Packages
   ##### REQUIRED TO RUN THE PROGRAM: 
   1. ***basemap***
       - coming soon
   2. ***haversine***
       - coming soon
   3. ***matplotlib***
       - coming soon
   4. ***pandas***
       - coming soon
   5. ***utils***
       - coming soon
   6. ***random***
       - coming soon
   7. ***argparse***
       - coming soon
   8. ***from sys import maxsize***
   9. ***from time import time***
   10. ***from time import time***
---
#### IV. Genetic Algorithm (GA)
A genetic algorithm is a heuristic search method used in artificial intelligent and computing. It is used for finding optimized solutions to search problems based on the theory of natural selection and evolutionary biology. Genetic algorithms are excellent for searching through large and complex data sets. They are considered capable of finding reasonable solutions to complex issues as they are highly capable of solving unconstrained and constrained optimization issues (Techopedia).

In a genetic algorithm, a population of candidate solutions (also called individuals, creatues, or phenotypes) to an optimization problem evolved toward better solutions (wikipedia). Each of these 'individual' solutions has a set of properties that can be manipulated or adjusted. Evolving from a population of randomly generated individuals, an iterative process, with the population in each iteration called a *generation*. After each generation is generated the fitness of every individual in the population is evaluated through a fitness function. The individuals that are more successful are selected from the current population, and each individual's attributes are then modified, or altered randomly, to form the next generation. Usually, GAs will terminate when either a max number of generations is reached, or a predetermined fitness level is achieved for the population.

The aim of this program is to find a more efficient solution than just brute force searching for an answer. While this particular algorithm is simple, understanding the behavior is difficult to understand. I struggled with the way GAs generate a solutions. 

However, genetic algorithms really only require a few basic steps:
   1. ***Initialization:***
       - Depending on the problem trying to be solved, the population size can vary. Often, the original population will be generated randomly, thus allowing for a wider range of possible mutations.
   2. ***Selection:***
       - After each generations a portion of the existing population is selected to start the next generation.
   3. ***Genetic Operators:****
       - These processes ultimately result in the next generation population of 'chromosomes' that is different from the initial generation.
       - The average fitness will have increase through this process for the overall population, since only the best individuals from the first generation are selected along with a small porportion of less fit solutions.
       - These less fit solutions ensure genetic diversity within the genetic pool of the parents and therefore ensure the genetic diversity of the subsequent generation of children.
   4. ***Termination:***
       - There are various ways to terminate once a condition has been reached. Wikipedia lists the most common:
           1. A solution is found that satisfies minimum criteria.
           2. Fixed number of generations reached
           3. Allocated budget (computation time/money) reached
           4. The highest ranking solution's fitness is reaching or has reached a plateau such that successive iterations are no longer producing better results.
           5. Manual inspection
           6. Combinations of the above

Limitations (Medium)
   1. Fitness Modelling is hard
   2. Dysfunctional for complex problems
   3. Unclear stop criteria
   4. Local optimum traps
   5. Limited use Cases 

---
#### V. Analysis of Performance

---
#### VI. Final Notes

---
#### VII. References
   1. **lccasagrande**
        - https://github.com/lccasagrande/TSP-GA
        - Majority of the code is based on Mr. lccasagrande's implementation.
   2. **Hands-On Machine Learning with Scikit-Learn and TensorFlow**
        - Various concepts of Neural Networks, evolutionary algorithms (EA).
   3. **City Locations**
        - https://www.latlong.net/place/
        - Used for all the latitude and Longitude values for each city.
   4. **Multi-stop Route Planner and Optimization Tools - Mapquest**
        - https://www.mapquest.com/routeplanner
        - Base Route = 15638 km
   5. **Techopedia**
        - https://www.techopedia.com/definition/17137/genetic-algorithm
        - Used to define Genetic Algorithms
   6. **Medium**
        - https://medium.com/@arbistarkillerllaveshi/genetic-algorithms-summary-limitations-3da8df3e1138
        - Used to describe the limitations of GA
   7. **Genetic Algorithms-Overview, Limitations and Solutions**
        - https://www.krishisanskriti.org/vol_image/04Jul201511073305%20%20%20%20%20%20%20Dayanand%205%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20329-333.pdf