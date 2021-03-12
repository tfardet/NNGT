import io
import os

import cartopy
from cartopy.io import Downloader


"""
Temporary fix for Cartopy loading NaturalEarth data until the following PR
is merge: https://github.com/SciTools/cartopy/pull/1745
"""

class NEShpDownloader(Downloader):
    '''
    Specialise :class:`cartopy.io.Downloader` to download the zipped
    Natural Earth shapefiles and extract them to the defined location
    (typically user configurable).

    The keys which should be passed through when using the ``format_dict``
    are typically ``category``, ``resolution`` and ``name``.
    '''
    FORMAT_KEYS = ('config', 'resolution', 'category', 'name')

    # Define the NaturalEarth URL template. The natural earth website
    # returns a 302 status if accessing directly, so we use the naciscdn
    # URL directly.
    _NE_URL_TEMPLATE = ('https://naciscdn.org/naturalearth/{resolution}'
                        '/{category}/ne_{resolution}_{name}.zip')

    def __init__(self, url_template=_NE_URL_TEMPLATE,
                 target_path_template=None, pre_downloaded_path_template=''):
        ''' adds some NE defaults to the __init__ of a Downloader'''
        super().__init__(url_template, target_path_template,
                         pre_downloaded_path_template)

    def zip_file_contents(self, format_dict):
        '''
        Return a generator of the filenames to be found in the downloaded
        natural earth zip file.

        '''
        for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg']:
            yield ('ne_{resolution}_{name}'
                   '{extension}'.format(extension=ext, **format_dict))

    def acquire_resource(self, target_path, format_dict):
        '''
        Download the zip file and extracts the files listed in
        :meth:`zip_file_contents` to the target path.

        '''
        from zipfile import ZipFile

        target_dir = os.path.dirname(target_path)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        url = self.url(format_dict)

        shapefile_online = self._urlopen(url)

        zfh = ZipFile(io.BytesIO(shapefile_online.read()), 'r')

        for member_path in self.zip_file_contents(format_dict):
            ext = os.path.splitext(member_path)[1]
            target = os.path.splitext(target_path)[0] + ext
            member = zfh.getinfo(member_path.replace(os.sep, '/'))
            with open(target, 'wb') as fh:
                fh.write(zfh.open(member).read())

        shapefile_online.close()
        zfh.close()

        return target_path

    @staticmethod
    def default_downloader():
        '''
        Return a generic, standard, NEShpDownloader instance.

        Typically, a user will not need to call this staticmethod.

        To find the path template of the NEShpDownloader:

            >>> ne_dnldr = NEShpDownloader.default_downloader()
            >>> print(ne_dnldr.target_path_template)
            {config[data_dir]}/shapefiles/natural_earth/{category}/\
ne_{resolution}_{name}.shp
        '''
        default_spec = ('shapefiles', 'natural_earth', '{category}',
                        'ne_{resolution}_{name}.shp')
        ne_path_template = os.path.join('{config[data_dir]}', *default_spec)
        pre_path_template = os.path.join('{config[pre_existing_data_dir]}',
                                         *default_spec)
        return NEShpDownloader(target_path_template=ne_path_template,
                               pre_downloaded_path_template=pre_path_template)

    
# add a generic Natural Earth shapefile downloader to the cartopy config 
# dictionary's 'downloaders' section.
_ne_key = ('shapefiles', 'natural_earth')
cartopy.config['downloaders'].setdefault(_ne_key,
                                         NEShpDownloader.default_downloader())


def natural_earth(resolution='110m', category='physical', name='coastline'):
    '''
    Return the path to the requested natural earth shapefile,
    downloading and unzipping if necessary.

    To identify valid components for this function, either browse
    NaturalEarthData.com, or if you know what you are looking for, go to
    https://github.com/nvkelso/natural-earth-vector/tree/master/zips to
    see the actual files which will be downloaded.

    Note
    ----
        Some of the Natural Earth shapefiles have special features which are
        described in the name. For example, the 110m resolution
        "admin_0_countries" data also has a sibling shapefile called
        "admin_0_countries_lakes" which excludes lakes in the country
        outlines. For details of what is available refer to the Natural Earth
        website, and look at the "download" link target to identify
        appropriate names.
    '''
    # get hold of the Downloader (typically a NEShpDownloader instance)
    # which we can then simply call its path method to get the appropriate
    # shapefile (it will download if necessary)
    ne_downloader = NEShpDownloader.default_downloader()
    format_dict = {'config': cartopy.config, 'category': category,
                   'name': name, 'resolution': resolution}
    return ne_downloader.path(format_dict)
                                         
