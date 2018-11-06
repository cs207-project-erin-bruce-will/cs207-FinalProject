# Milestone 2

**Due: Thursday, November 8th at 11:59 PM**

## Submission Instructions
You will be writing code for this milestone.  Please be sure that we can find it in your project
organization.  

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

## Requirements
Here are the main requirements for the second milestone:
1. Working forward mode implementation
   * See the sections below for more specific details
2. Test suite
3. Updated / extended documentation
4. Proposal for additional features
(Support for arrays, add dot product function, etc)
(Jacobian function, "AD.Jacobian". Returns matrix of derviatives)

### Working Forward Mode Implementation
You must have a working forward mode implementation.  This does not mean, however, that you should
have a complete implementation.  Here are guidelines.

#### Minimum Package Requirements
* The software should be available for download from your GitHub organization.  
* At this point, there is no need for you to release it on PyPI.  
* You should provide a `requirements.txt` file with your software so other developers are able to
  install the necessary dependencies.
* After a user installs your package, they should be able to use it without difficulty.

#### Minimum Implementation Requirements
The following is a description of a typical use case.

* A user downloads your package from your organization (`pip` not required yet).
* They install the dependencies and run the tests.
* They create a "driver" script in the top level.
  - Note:  How they interact with your package will depend on your implementation.  The interface
    and other implementation details should be described in your documentation.
  - The next few steps may sound somewhat abstract, but that is only because they hinge on your
    specific implementation.
* In the driver script, they import your package.
* They instantiate an automatic differentiation object to be used in the foward mode.
* They write a root-finding algorithm (e.g. Newton's method) that requires calculation of the
  Jacobian.
* They access the Jacobian via the automatic differentiation object.

There are other use cases as well.  Someone may want to solve an optimization problem using the
forward mode.  Regardless, the workflow outlined above should be roughly the same.

#### What Kinds of Functions should be Implemented?
This document began by saying that you don't need to have the forward mode fully implemented.
This section describes what that means.

Recall that eventually a user should be able to use your software to get derivatives of a vector
function of a vector.  You should keep this in mind when designing your software.  For this
milestone, however, the requirements are a little bit relaxed.  A typical user this time will just
want to calculate derivatives of a scalar function of a scalar.  At a minimum, your software should
overload all the basic operations (addition, multiplication, subtraction, division, power).  Don't
forget about the unary operations such as negation.  It should also contain the following elemental
functions:
* exponential
* trig functions (sine and cosine)

Feel free to go beyond the bare minimum.  You'll have to do it eventually before the semester ends.


### Test Suite
You should have a test suite that runs with `pytest`.  Your test suite should run automatically on
Travis CI.  The project GitHub repo should contain a badge showing the pass/fail status of your build.

You should also have your project connected to Coveralls.  Once again, the project repo should have
a badge reporting on the coverage of your code from Coveralls.

### Documentation
Be sure to extend your documentation from Milestone 1.  Now is a good time to think about the best
way to distribute your documentation.  You may want to consider using a Jupyter notebook with
Markdown cells interspersed with code blocks for actual hands-on demos.  The same thing can be
achieved with a standard Markdown document, but it won't be as interactive.

You will receive full points as long as you have a documentation folder and your documentation is
complete.  However, you may want to consider alternative ways of hosting your documentation.  For
example:  [Read the Docs](https://readthedocs.org/) or
[Sphinx](https://docs.readthedocs.io/en/latest/intro/getting-started-with-sphinx.html).

#### Documentation Sections
The following sections should be present:
* Introduction
  - Describe problem the software solves and why it's important to solve that problem.  This can be
    the same as Milestone 1.
* How to use your package
  - How to install?  Even (especially) if the package isn't on `PyPI`, you should walk them through
    the creation of a virtual environment or some other kind of manual installation.
  - Include a basic demo for the user.  Come up with a simple function to differentiate and walk the
    user through the steps needed to accomplish that task.
* Background
  - This can be the same as in Milestone 1.
* Software organization
  - High-level overview of how the software is organized.
    * Directory structure
    * Basic modules and what they do
    * Where do tests live?  How are they run?  How are they integrated?
    * How can someone install your package?  At this point, it is okay if your package isn't on
      `PyPI`.  If it's not, then you should describe how someone can download and install your
      package manually.
* Implementation details
  - Description of current implementation.  This section goes deeper than the high level software
    organization section.
    * Try to think about the following:
      - Core data structures
      - Core classes
      - Important attributes
      - External dependencies
      - Elementary functions
    This is similar to what you did for milestone 1, but now you've actually implemented it.
  - What aspects have you not implemented yet?  What else do you plan on implementing?

### Future
Now that you've got most of the hard implementation work done, what kinds of things do you want to
impelement next?  How will your software change?  What will be the primary challenges?
