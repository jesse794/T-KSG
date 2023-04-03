# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
sys.path.append("../src")
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random
import MLP
import pyffx


def truncate(number, digits) -> float:
    """truncate an incoming float to a givin number of places beyond the
    decimal."""
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def encFloat(floatValInput, direction):
    """format preserving encryption/decrytion of float by temporarily
    transforming it to an int.

    N.B.  This entire function is pretty kludgey to care of cases where
    the encrypted or decrypted value either has leading or trailing
    zeros.  I think the kludges will only work if the input is >=1.0 and
    <10.  If you wanted it to work for larger values, you'd have to: a)
    change lengthOfInt from 9 to 10, b) fixup some other problems that
    would cause in the code"""

    # confirm that incoming value (for encryption case) is >=1.0 and
    # <10.  To handle values outside that range would require
    # modifications to this routine.
    if (direction == 1 and (not (floatValInput >= 1.0 and floatValInput < 10.0))):
        print('input value for encryption outside of allowed range')
        sys.exit()

    # scale the incoming float
    scaleFactor = 1e8
    intConvertedValue = int(round((floatValInput * scaleFactor), 0))
    # kludge to take care of case where input has trailing zero and so
    # would get scaled up by 10.
    if (intConvertedValue % 10 == 0 and direction == 1):
        intConvertedValue = int(intConvertedValue / 10)

    # setup encryption.
    lengthOfInt = 9
    ec = pyffx.Integer(b'secret-key', length=lengthOfInt)

    # do the encryption/decryption
    if direction == 1:  # encrypt
        transformedInt = ec.encrypt(intConvertedValue)
    elif direction == 2:  # decrypt
        transformedInt = ec.decrypt(intConvertedValue)
    else:  # bad input
        print ("bad input value")
        sys.exit()
    # print(transformedInt)

    finalResult = float(transformedInt / scaleFactor)
    # take care of case where output was shorter int and so gets cut by
    # factor of ten when converting back to float.
    if (finalResult < 1 and direction == 2):
        finalResult *= 10

    return finalResult


def encryptionTest(nEntries):
    """
    Check that round-trip encrypt-decrypt puts all entries in list back
    to original values
    """
    # Make random list
    x = [np.random.uniform(5.0, 6.0, 1)[0] for i in range(nEntries)]
    # define encryption function and encrypt the list
    ec = pyffx.Integer(b'secret-key', length=4)
    x_enc = [encFloat(i, 1) for i in x]
    # now decrypt the encrypted list
    x_dec = [encFloat(i, 2) for i in x_enc]
    # now get the diff between original and encrypted lists entry by
    # entry
    enc_dec_diff = [x[i] - x_dec[i] for i in range(len(x_enc))]

    # max value of the entry-by-entry diff should be zero, or at least
    # should be pretty small (rounding will likely prevent it from being
    # zero)
    return max(enc_dec_diff)


# +
# number of each event type to generate
nEvents = 10000

# Generate signal and background events, and fill answer arrays
vars_sig = [[np.random.uniform(5.0, 6.0, 1)[0]] for i in range(nEvents)]
ans_sig = [[1.0] for i in range(nEvents)]
vars_bkd = [[np.random.uniform(6.2, 7.2, 1)[0]] for i in range(nEvents)]
ans_bkd = [[-1.0] for i in range(nEvents)]

# merge sig and bkd events into single lists
vars_merged = np.concatenate((vars_sig, vars_bkd), axis=0)
ans_merged = np.concatenate((ans_sig, ans_bkd), axis=0)

# truncate to fixed precision
vars_mergedTrunc = [[truncate(i[0], 8)] for i in vars_merged]

# compute MI
originalMI = MLP.mi_binary(vars_mergedTrunc, ans_merged, 1, 2)

# encrypt the vars
vars_enc = [[encFloat(i[0], 1)] for i in vars_mergedTrunc]

# decrypt the encrypted list so you can check that MI retuerns to
# original value
vars_dec = [[encFloat(i[0], 2)] for i in vars_enc]

# compute MI using encrypted vars
encryptedMI = MLP.mi_binary(vars_enc, ans_merged, 1, 2)

# compute MI using decrypted vars to confirm that it returns to original
# value
decryptedMI = MLP.mi_binary(vars_dec, ans_merged, 1, 2)

print("running check to confirm that encryption routine is working")
print("results of encryption round-trip check (should be << 1.0)=", encryptionTest(10))
print("computing MI before encryption, after encryption, and after decryption")
print("originalMI=", originalMI, "encryptedMI=", encryptedMI, "decryptedMI=", decryptedMI)
