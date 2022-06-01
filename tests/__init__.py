import numpy as np

def compare_array_elementwise(testcase,ref,res,print_vals=True):
    testcase.assertEqual(ref.shape,res.shape)
    if print_vals:
        for i in range(ref.shape[0]):
            for j in range(ref.shape[1]):
                if not np.isclose(ref[i, j] , res[i, j]):
                    print("{},{}: ref: {},res:{}".format(i,j,ref[i,j],res[i,j]))
    testcase.assertTrue(np.allclose(ref,res))