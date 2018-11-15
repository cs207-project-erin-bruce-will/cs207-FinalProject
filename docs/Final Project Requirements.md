# Final Deliverables

**Due: Thursday, November 12th at 9:00 AM**

## Submission Instructions
Your project should be available on in your GitHub organization though your project repo.

Your submission should be in the following format:
```
project_repo/
             README.md
             docs/  
                  milestone1
             code/
                 ...
```
Note that `code` should be named something informative.

## Software Requirements
Here are the main requirements for the final project:
1. Working forward mode implementation
   * See the sections below for more specific details
2. Test suite
3. Updated / extended documentation
4. New features

### Working Forward Mode Implementation
You must have a working forward mode implementation.  Your library should be able to be used on real
functions of one more more variable.

#### Minimum Package Requirements
* The software should be available for download from your GitHub organization.  
* The software should be installable from PyPI.
* You should provide a `requirements.txt` file with your software so other developers are able to
  install the necessary dependencies.
* After a user installs your package, they should be able to use it without difficulty.

#### Minimum Implementation Requirements
The following is a description of a typical use case.

* A user downloads your package from your organization or through PyPI.
* They install the dependencies.
* They run the tests if they're a fellow developer.
* They create a "driver" script in the top level.
  - Note:  How they interact with your package will depend on your implementation.  The interface
    and other implementation details should be described in your documentation.
  - The next few steps may sound somewhat abstract, but that is only because they hinge on your
    specific implementation.
* In the driver script, they import your package.
* They instantiate an automatic differentiation object to be used in the foward mode.
* They use the automatic differentiation objects in their own applciations (root-finding,
    optimzation, etc).

#### What Kinds of Functions should be Implemented?
All elementary functions and basic operations should be implemented.

##### Basic Operations
* Addition (commutative)
* Subtraction
* Multiplication (commutative)
* Division
* Power
* Negation

##### Comparison Operators
It is up to you which comparison operators to implement.  Here are some options:
* `__lt__` (less than)
* `__gt__` (greater than)
* `__le__` (less than or equal to)
* `__ge__` (greater than or equal to)
* `__eq__` (equal to)
* `__ne__` (not equal to)

At the very least, it makes sense to have `__eq__` and `__ne__`.  I will be interested to see what
(if any) uses you find for the other operators.

##### Elementary Functions
* Trig functions (at the very least, must have sine, cosine, tangent)
* Inverse trig functions (e.g. arcsine, arccosine, arctangent)
* Exponentials
  - Should be able to handle any base
  - You can treat the natural base (e) as a special case
    * This is what `numpy` does
* Hyperbolic functions (sinh, cosh, tanh)
  - Note that these can be formed from the natural exponential (e)
  - Therefore, it is up to you whether or not to include them as elementary functions
* Logistic function
  - Again, this can be formed from the natural exponential
* Logarithms
  - Should be able to handle any base
* Square root


### Test Suite
You should have a test suite that runs with `pytest`.  Your test suite should run automatically on
Travis CI.  The project GitHub repo should contain a badge showing the pass/fail status of your build.

You should also have your project connected to Coveralls.  Once again, the project repo should have
a badge reporting on the coverage of your code from Coveralls.

### Documentation
Your documentation must be complete, easy to navigate, and clear.  Call your the final form of your
documentation `documentation`.

Your documentation should be a mix of text and hands-on demos.  As always, it is up to you and your
group on the best way to accomplish this (e.g. Jupyter notebook, GitHub README, Sphinx/Read the
Docs).

You will receive full points as long as you have a documentation folder and your documentation is
complete.  However, you may want to consider alternative ways of hosting your documentation.  For
example:  [Read the Docs](https://readthedocs.org/) or
[Sphinx](https://docs.readthedocs.io/en/latest/intro/getting-started-with-sphinx.html).

#### Documentation Sections
The following sections should be present:
* Introduction
  - Describe problem the software solves and why it's important to solve that problem.  This can be
    built off of the milestones, but you may need to update it depending on what new feature you
proposed.
* How to use your package
  - How to install?
  - Include a basic demo for the user. This can be based off of the milestone, but it may change
    depending on what your new feature is.  
    * You may want to consider more than one basic demo: one demo just for automatic differentiation
      and and one demo for your new feature.
    * Note that this is very much dependent on your final deliverables!
    * Keep the basic demos to a manageable number.
* Background
  - The automatic differentiation background can probably stay the same as in the milestones, unless
    you were told to update it considerably.
  - Be sure to include any necessary background for your new feature.
* Software organization
  - High-level overview of how the software is organized.
    * Directory structure
    * Basic modules and what they do
    * Where do tests live?  How are they run?  How are they integrated?
    * How can someone install your package?  Should developers and consumers follow a different
      installation procedure?
* Implementation details
  - Description of current implementation.  This section goes deeper than the high level software
    organization section.
    * Try to think about the following:
      - Core data structures
      - Core classes
      - Important attributes
      - External dependencies
      - Elementary functions
  - What aspects have you not implemented yet?  What else do you plan on implementing?  There is
    always more to do, even if you satisfied all the project requirements.  You probably have some
    `NotImplementedError` exceptions in some places for future work.

### Future
What else do you want to add?  What's missing?  Don't just think about mathematical things here.
Try to think about applications that you'd like to have use your code.  Just about every area of
science can use automatic differentiation (physics, biology, genetics, applied mathematics,
optimization, statistics / machine learning, health science, etc).


## Presentation Requirements
* Each group will have **15 minutes** to present your package.
  - 12 minutes for the actual presentation
  - 3 minutes for questions
  - * **Do not** go over the allotted time; you will be cut off.

### Things to keep in mind
* Remember, the teaching staff already has full access to your code.
  - No need to present actual code in your talk
* Pseudo-code and flowcharts can be very useful to give the big idea of how your package works.
* Library demos can be very useful, but be careful.  If they don't work well then you'll waste all
   your presentation time.
* All group members should present during the allotted time.
* All group members should be prepared to answer questions about any portion of the project (not
  just the part they present on)
* You should provide sufficient background for the project.
  - Don't overdo the mathematical details for automatic differentation.  We are already familiar
    with them.
  - Instead, provide the big ideas behind automatic differentation and the motivation for using it.
* Spend a fair bit of time on your new feature.
  - You may need to present some mathematical background to get your audience oriented.
* Be sure to conclude with future work and possible extensions.
