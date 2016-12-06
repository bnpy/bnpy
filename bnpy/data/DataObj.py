from DataIterator import DataIterator

class DataObj(object):

    ''' Abstract base class for all bnpy data objects.

    Defines all required functions that base classes must inherit.
    '''

    @classmethod
    def read_npz(self, npzfilepath, **kwargs):
        ''' Constructor for building data object from .npz file
        '''
        pass

    @classmethod
    def read_mat(self, matfilepath, **kwargs):
        ''' Constructor for building data object from .mat file
        '''
        pass

    def __init__(self, *args, **kwargs):
        ''' Constructor for building data object in memory.
        '''
        pass

    def to_iterator(self, **kwargs):
        ''' Create iterator for processing subsets of this dataset.

        Returns
        ------
        DI : bnpy.data.DataIterator.
        '''
        return DataIterator(self, **kwargs)

    def get_short_name(self):
        ''' Returns human-readable name viable for system file paths.

        Useful for creating filepaths specific for this data object.

        Returns
        -------
        name : string
        '''
        if hasattr(self, 'name'):
            return self.name
        return "UnknownData"

    def get_text_summary(self, **kwargs):
        ''' Returns human-readable description of this dataset.

        Summary might describe source of this dataset, author, etc.

        Returns
        -------
        summary : string
        '''
        s = 'DataType: %s. Size: %d' % (
            self.__class__.__name__, self.get_size())
        return s

    def get_num_examples(self, **kwargs):
        ''' Get count of active, in-memory units for this dataset.

        Returns
        -------
        size : int
        '''
        pass

    def get_total_num_examples(self, **kwargs):
        ''' Get count of all units associated with this dataset.

        Returns
        -------
        size : int
        '''
        pass

    def make_subset(self, unitIDs, **kwargs):
        ''' Get subset of this dataset identified by provided unit IDs.

        Returns
        -------
        Dchunk : bnpy.data.DataObj subclass
        '''
        pass

    def add_data(self, DataObj):
        ''' Appends (in-place) provided dataset to this dataset.

        Post Condition
        -------
        self grows by adding all units from provided DataObj.
        '''
        pass
