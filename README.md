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
- investigate dynamic initialization/warm start?
- support distributed data on spark