0x02. Calculus
==============

Learning Objectives
-------------------

### General

-   Summation and Product notation
-   What is a series?
-   Common series
-   What is a derivative?
-   What is the product rule?
-   What is the chain rule?
-   Common derivative rules
-   What is a partial derivative?
-   What is an indefinite integral?
-   What is a definite integral?
-   What is a double integral?

Tasks

9. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities
Write a function def summation_i_squared(n): that calculates \sum_{i=1}^{n} i^2:

- n is the stopping condition
- Return the integer value of the sum
- If n is not a valid number, return None
- You are not allowed to use any loops

10. Derive happiness in oneself from a good day's work
Write a function def poly_derivative(poly): that calculates the derivative of a polynomial:

- poly is a list of coefficients representing a polynomial
  - the index of the list represents the power of x that the coefficient belongs to
  - Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
- If poly is not valid, return None
- If the derivative is 0, return [0]
- Return a new list of coefficients representing the derivative of the polynomial

17. Integrate
Write a function def poly_integral(poly, C=0): that calculates the integral of a polynomial:

- poly is a list of coefficients representing a polynomial
  - the index of the list represents the power of x that the coefficient belongs to
  - Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
- C is an integer representing the integration constant
- If a coefficient is a whole number, it should be represented as an integer
- If poly or C are not valid, return None
- Return a new list of coefficients representing the integral of the polynomial
- The returned list should be as small as possible
