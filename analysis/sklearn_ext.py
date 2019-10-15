from sklearn.base import BaseEstimator, TransformerMixin

class DummyTransformer(BaseEstimator, TransformerMixin):
    """
    Transfomer that just returns the data back and 
    preservces the column names if the X is a dataframe.

    Can be used in the 'remainder' parameter in ColumnTransformer
    instead of 'passthrough', which loses the column names.

    Can also be used in ColumnTransformer to select a set of
    columns and then set the remainder='drop'
    """

    def fit(self, X, y=None):
        """
        Overwrites base fit method and saves columns names 
        if X is a dataframe to object property

        Parameters
        ----------
        X : 2-d matrix or dataframe
        y : 1-d list

        Returns
        -------
        The object itself
        """

        try:
            self.columns = X.columns
        except:
            pass
        return self
    
    def transform(self, X):
        """
        Overwrites base transform method.
        No transformation is done, returns X back

        Parameter
        ---------
        X : 2-d matrix or dataframe

        Returns
        -------
        X
        """

        return X
    
    def get_feature_names(self):
        """
        Retuns column names back so that get_feature_names 
        method can be called with ColumnTransformer

        Returns
        -------
        List of column names saved when fit() was called
        """

        return self.columns