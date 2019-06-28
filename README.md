# matrixFactorizer
Matrix factorization optimizer, for applications like recommendation system, simple embedding/featurization. This class is useful when in reality we only store elements where there is a known data point, instead of the whole matrix.


Clone repository and install
```
$ python setup.py install
```

In python script, call package like:
```
import matrixFactorizer
mf = matrixFactorizer.MatrixFactorizer(10,20,3,0.1)
```

package WIP

TODO:
- add test case
- detect divergence
- use input lr as initial learn rate, add some simple line search
- investigate dynamic initialization/warm start?
- necessary to add L1 penalty?
- add 2nd order optimization (coordinate descent with exact Hessian) as an option
- support distributed data on spasrk