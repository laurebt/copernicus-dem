def get_summary(pre_defined_shape):
    
    s = '''
    
    Retrieve this DEM by clicking on "Render DEM" and then "Download" in the left hand side panel.  
    Alternatively, retrieve it from a Jupyter notebook:

    ```python

    import pydaisi as pyd
    copernicus_dem = pyd.Daisi("laiglejm/Copernicus DEM download")

    copernicus_dem.retrieve_dem(pre_defined_shape='''
    
    s += f"{pre_defined_shape}"
    s += ''', resolution="90", return_type="dem").value
    '''

    return s