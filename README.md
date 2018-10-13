# Milestone 1

**Due: Thursday, October 18th at 11:59 PM**

## Submission Instructions
In the homework assignment this week, you should have created a GitHub organization with your
project group and invited the teaching staff.  The organization can consist of multiple
repositories, but one of those repositories must be your actual software project repo.  That 
repo is the one that will be graded for your final project and the project milestones.

Within your final project repo, you should create a directory called `docs`.  You can use this
directory to organize documentation and tutorials for your final package.  For this milestone, you
should have a file called `milestone1`.  The type of file is up to you and your group.  Two good 
choices are markdown (`milestone1.md`) or a Jupyter notebook (`milestone1.ipynb`).

To summarize, your submission should be in the following format:
```
project_repo/
             README.md
             docs/  
                  milestone1
             ...
```

**The teaching staff will only be able to give you a grade if you follow the exact structure just
outlined!**

## Requirements
There are three primary requirements for this first milestone.
1. Create project organization and invite teaching staff.
   * Within the project organization, create a project repo (make sure teaching staff has access).
2. Create a `README.md` inside the project repo.  At this point, the `README` should only include
the group name / number and list the members of the team.
3. The `docs/` directory should include a document called `milestone1` (the extension is up to you,
but `.md` or `.ipynb` are recommended.  Details on how to create `milestone1` are provided in the
`Milestone1` section below.

## Milestone1 Document
You must clearly outline your software design for the project.  Here are some possible sections to
include in your document along with some prompts that you may want to address.





### Introduction
TODO:Erin
Describe problem the software solves and why it's important to solve that problem

### Background
TODO:Erin
Describe (briefly) the mathematical background and concepts as you see fit.  You **do not** need to
give a treatise on automatic differentation or dual numbers.  Just give the essential ideas (e.g.
the chain rule, the graph structure of calculations, elementary functions, etc).

### How to Use *PackageName*

Example usage (option A):
```
import autodif as ad
def my_fun(x):
	ad.sin(x**2)

dual = ad.DualNumber(7)
value_at_7, derivative_at_7 = myfun(dual)
value_at_8, derivative_at_8 = myfun(dual.set(8))

# evaluating and differentiating a transient function
value_at_8, derivative_at_8 = dual+cos(dual)
```

Example usage (option B):
```
import autodif as ad
x = ad.Variable('x')
y = ad.Variable('y')
my_fun = ad.sin(x*y)

my_fun.compute({x:7,y:3})
# returns 1) scalar output 2) vector-like d_out/d_in_1, d_out/d_in_2

my_vec_fun = ad.VectorOutput(my_fun+x, ad.sin(y)/x)
my_vec_fun.compute({x:7,y:3})
# returns 1) vector output 2) matrix-like d_out_1/d_in_1, d_out_1/d_in_2; d_out_2/d_in_1, d_out_2/d_in_2
```
Considerations: 
1) We could instead take in seed vectors, e.g. (2,3) and return the slope along that vector instead of returning the full Jacobian.
2) We're not certain option A will support multiple inputs to a function, e.g. my_fun(x,y) = x\*y. It's not clear how a dual number can hold two derivatives neatly. Do we end up with $(x_r + \epsilon x_e)(y_r + \epsilon y_e) = (x_ry_r + \epsilon x_ey_r + y_ex_r)$? Would $(3 + \epsilon 6)(2 + \epsilon 5) = (6 + \epsilon 27)$, or do we need to keep $(6 + \epsilon 12+15)$. If the latter, aren't we building very large and complex post-epsilon components that are more or less symbolic derivatives?

Is the answer to treat the epsilons in the expression as actually $\esplion_x$ and $\espilon_y$ so that they can't directly add, but still multiply to 0? I think that works, but it feels exactly the same as having a slot for the value of del_x and a slot for the value of del_y. Ah! So we could have dual numbers of the form (real, eps_1, eps_2, eps_3, ...) where each eps is a different variable. But each dual number needs to track which eps it has (is position 3 in this expression x or y or z?). 

Multiplying, adding, and dividing would be a matter of producing a new dual number with appropriate real, eps and name_dict.

So we'd ultimately have a dual number that looks like:

```
class DualNumber:
    self.real # the real-number component
    self.eps #list of real numbers, encodes the epsilon for each variable that affects this node
    self.name_dict #maps a variable name to position in the list above; if not found derivative is 0
    
    #multiplication implementation
    def __mult__(self, other):
        output.real = self.real*other.real
        
        # suppose that self has 3 espilons and other has 2
        
        # real part of first parent distributes
        for k2 in other.eps:
            output[k2] += self.real*other.eps[self.name_dict[k2]]
        
        # real part of the second parent distributes
        for k1 in self.eps:
            # use a defaultdict; will automatically grow any epsilons that already exist
            # and create entries for ones that don't
            output[k1] += other.real*self.eps[self.name_dict[k1]]
            
        # needs some handling to build the output's name_dict. 
        # for the above, should just be keys = other.name_dict + self.name_dict (combine keys, drop duplicates)
        # values = range(len(keys))
                
```
$(a_r + \epsilon_x a_x +\epsilon_z a_z )(b_r + \epsilon_y b_y + \epsilon_x b_x) = (a_rb_r + \epsilon_y a_rb_y + \epsilon_x a_rb_x) + (\epsilon_x a_xb_r + 0 + 0) + (\epsilon_z a_zb_r + 0 + 0)$

$$=\left(a_rb_r + \epsilon_y a_rb_y + \epsilon_x (a_rb_x + a_xb_r)\right + \epsilon_z a_zb_r)$$


```
class ComputeNode:
    self.parents #list of ComputeNodes; the inputs to this node
    
    def compute(self) 
    # calls compute() on the parents to get thier real part and a dictionary of derivatives from each
    # that is, parent1 returns the value of the function and a dictionary like {x:5, y:-3} which are the
    # derivatives of the parent at the point of interest w.r.t x and w.r.t y
    #
    # The rest of compute() is applying the math formula for this node type. So a multiplicatin node would do:
    # real1, deriv_dict1 = parent1.compute()
    # real2, deriv_dict2 = parent1.compute()
    # real = real1*real2
    # deriv_x = real1*deriv_dict2[x] + real2*deriv_dict1[x]
    # deriv_y = real1*deriv_dict2[y] + real2*deriv_dict1[y]
    # ... (if we have more variables w,z, and so on.
    # return (real, {'x':deriv_x, 'y':deriv_y, ...}
```

Overall, there feel *VERY* similar. The first version is more eager-evaluation and always returning a latest-and-greatest (multi-)dual number; the second version is in some sense building the computation graph by tracking parents. The second version can respond naturally to changes in the graph, while the first version requires us to 

### Software Organization
Discuss how you plan on organizing your software package.
* What will the directory structure look like?  
* What modules do you plan on including?  What is their basic functionality?
* Where will your test suite live?  Will you use `TravisCI`? `Coveralls`?
* How will you distribute your package (e.g. `PyPI`)?

### Implementation
Discuss how you plan on implementing the forward mode of automatic differentiation.
* What are the core data structures?
* What classes will you implement?
* What method and name attributes will your classes have?
* What external dependencies will you rely on?
* How will you deal with elementary functions like `sin` and `exp`?

Be sure to consider a variety of use cases.  For example, don't limit your design to scalar
functions of scalar values.  Make sure you can handle the situations of vector functions of vectors and
scalar functions of vectors.  Don't forget that people will want to use your library in algorithms
like Newton's method (among others).

Try to keep your report to a reasonable length.  It will form the core of your documentation, so you
want it to be a length that someone will actually want to read.

## Additional Comments
There is no need to have an implementation started for this Milestone.  You are now in the planning
phase.  This means that you should feel free to have a `project_planning` repo in your project
organization for scratch work and code.  The actual implementation will start after Milestone 1.
