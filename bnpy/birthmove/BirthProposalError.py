'''
BirthProposalError.py

Simple wrapper around ValueError for birth-related exceptions
'''


class BirthProposalError(ValueError):

    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)
