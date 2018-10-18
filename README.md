[![Build Status](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject.svg?branch=master)](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject)

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
Automatic differentiation is a set of techniques that allows a computer program to numerically evaluate the derivative of a function. Automatic differentiation is a powerful tool that does not depend on the dimensionality of the predictor variable, and does not suffer from rount off errors. When working with large sets of data, manual calculation is unrealistic, and numeric differentiation requires a large volume of complicated code.

Dual numbers are an extension of a real number into two-dimesional space using two real numbers and a vector, epsilon. Dual numbers are useful for geometrical treatments, especially when considering analytical methods in kinematics and dynamics of spatial mechanisms. The application of dual numbers in science and engineering research is fairly common, and there is a need for an application that can differentiate them quickly and easily.

### Background
Automatic differentiation (AD) relies on the chain rule to perform a series of basic operations when computing the derivative of a function. Elementary arithemetic operations (such as add, subtract, multiply, divide, etc.) and elementary functions (such as exp, sine, cosine, etc.) are performed at each node, and AD provides the instruction to take the derivate of the given elementary operation or function. These derivatives are accumulated via the chain rule to return the full derivative of a function. There are two modes employed: forward mode, which performs differentiation based on the independent or predictor variable, and reverse mode, which performs differentiation based on the dependent or response variable. Automatic differentiation with dual numbers provides an additional wrinkle, as we must now account for the epsilon (vector) term.

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


**------------------------------Bruce Edit *to distinguish from Will's part in the same section*--------------------------**<BR>
**How to Use *PackageName***<BR>
A user will give a functional form, e.g. $f(x)=sin(x^2)+\frac{1}{2}x\cdot cos^2(x)+3x^3$. She is interested in knowing the functional value and derivative at $x=3$. This complex function has to be in some format that we can recognize. We may provide user a template to fill in the functional form. Also we may accept multiple variables, multiple functional forms simutaneously and/or find higher-order derivatives. In addition, the user needs to provide point(s) they are interested in evaluating the functional value(s) and the corresponding derivative value(s).<BR><BR>
Once the functional form is provided by the user, our software creates dual number class object, parse the functional form into pieces that can be attributed to any of the fundamental functional forms. To guarantee this works, we need to create all possible fundamental functional form:<BR>

* $\sin(x)$, $\cos(x)$, and $\tan(x)=\frac{\sin(x)}{\cos(x)}, etc.$<BR>
* $x^r$, where $r>0$ or $\frac{1}{x^r}$<BR>
* $\log_bx$, where b is the base of logarithm function<BR>
* $a^x$, where $a>0$<BR>
* Any combination of these functional forms<BR>

Ideally, we design an interface that serves as the only channel to interact with users. This may lead us to also include some GUI class for visual interface.<BR>

Take the above functional form as an example, our dual number creates an object with at least two attributes: `.value` and `.derivative`. Our parsing class should decompose the above function following the table below:<BR>

| Trace | Elementary Function | Current Value | Elementary Function<BR> Derivative | Elementary Function Derivative Value | 
| :---: | :-----------------: | :-----------: | :----------------------------: | :--------------------------------------------------------: | 
| $x_{1}$ | $x_{1}$ | 3 | $\dot{x}_{1}$ | $1$ |
| $x_{2}$ | $x_{1}^2$ | $9$ | $2x_{1}\dot{x}_{1}$ | $6$ |
| $x_{3}$ | $\sin(x_{2})$ | $\sin(9)$ | $\cos(x_{2})\dot{x}_{2}$ | $6\cos(9)$ |
| $x_{4}$ | $\frac{x_{1}}{2}$ | $\frac{3}{2}$ | $\frac{\dot{x}_{1}}{2}$ | $\frac{1}{2}$ |
| $x_{5}$ | $\cos(x_{1})$ | $\cos(3)$ | $-\sin(x_{1})\dot{x}_{1}$ | $-\sin(3)$ |
| $x_{6}$ | $x_{5}^2$ | $\cos^2(3)$ | $2x_{5}\dot{x}_{5}$ | $-2\sin(3)\cos(3)$ | 
| $x_{7}$ | $x_4\cdot x_{6}$ | $\frac{3}{2}\cos^2(3)$ | $x_4\dot{x}_{6}+\dot{x}_{4}x_6$ | $\frac{1}{2}\cdot \cos^2(3)-3\sin(3)\cos(3)$ |
| $x_{8}$ | $3x_1^3$ | $81$ | $9x_1^2\cdot \dot{x_1}$ | $81$ |
| $x_{9}$ | $x_3+x_7+x_8$ | $\sin(9)+\frac{3}{2}\cos^2(3)+81$ | $\dot{x}_3+\dot{x}_7+\dot{x}_8$ | $6\cos(9)+\frac{1}{2}\cos^2(3)-3\sin(3)\cos(3)+81$ |
	

User wants to evaluate derivatives at $a=3$, and provide a functional form $f(x)=sin(x^2)+\frac{1}{2}x\cdot cos^2(x)+3x^3$. Then inside our software, an automatic parser `separ` decomposes the function into pieces according to the add/minus sign:

$$f_1,f_2,f_3 = separ(f_x)$$
    
For $f_1, f_2, f_3$, a dual number object is created:

$$x = myAD(a)$$

For each piece (i.e. $f_1,f_2,f_3$, etc.), second parser `myParser` create layers. For example, the second piece $f_2$ should be further parsed as $y_1 = \frac{1}{2}x_1$, $y_2 = \cos(x_1)$, and $y_3 = y_2^2$:

$$y1,y2,y3 = myParser(f2)$$

Now evaluate, for instance the functional value and derivative for piece $f_2$, where $x$ is a `myAD` object:
$$f_2=\frac{1}{2}x cos^2(x)$$

**----------------------------------------------End of Bruce Edit-------------------------------------------**<BR>


### Software Organization
Discuss how you plan on organizing your software package.<BR>
* What will the directory structure look like?
    - The following will be our preliminary directory structure:
    ```
project_repo/
             README.md
             docs/  
                  milestone1
                  milestone2
                  milestone3
                  technical documentation
                  user's guide
                  ...
             package_deliverable/
                  class1.py
                  class2.py
                  class3.py
                  others.py
                  ...
             test/
                  
             ...
```
* What modules do you plan on including? What is their basic functionality?
    - Based on the prior section, we plan on delivering the following modules:
        * `separ`: a preliminary parsing class to break down user's formatted inputs into pieces. We can separate by addition/subtraction or some other basic operation (e.g. log, exp, etc.)
        * `myAD`: dual number class and methods that overload operators, etc.
        * `myParser`: further decomposes into smaller, fundamental functions that can be captured and recognized by our all types of overloaded operator rules for different fundamental functions (e.g. $\left(e^{x}\right)' = e^{x}$; $ln'(x) = \frac{1}{x}$, etc.).
    - In addition, we may include some GUI modules and/or I/O modules - more details will follow in milestone 2 document.
* Where will your test suite live? Will you use TravisCI? Coveralls?
    - We will perform all our tests in the *test* folder. We will create virtual environment for individual test.
    - Yes, we've set up TravisCI and Coveralls for our project.

* How will you distribute your package (e.g. PyPI)
    - Yes, ...
    - 
    

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
