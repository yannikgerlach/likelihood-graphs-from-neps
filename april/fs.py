# Copyright 2019 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

from pathlib import Path

import arrow

# Base
ROOT_DIR = Path(__file__).parent.parent

# Base directories
OUT_DIR = ROOT_DIR / '.out'  # For anything that is being generated
RES_DIR = ROOT_DIR / '.res'  # For resources shipped with the repository
CACHE_DIR = OUT_DIR / '.cache'  # Used to cache event logs, results, etc.

# Resources
PROCESS_MODEL_DIR = RES_DIR / 'process_models'  # Randomly generated process models from PLG2
BPIC_DIR = RES_DIR / 'bpic'  # BPIC logs in XES format

# Output
EVENTLOG_DIR = OUT_DIR / 'eventlogs'  # For generated event logs
MODEL_DIR = OUT_DIR / 'models'  # For anomaly detection models
PLOT_DIR = OUT_DIR / 'plots'  # For plots
TENSORBOARD_DIR = OUT_DIR / 'tensorboard'
CONFNET_DIR = OUT_DIR / 'confnet'  # For alignment models
EVALUATION_DIR = OUT_DIR / 'evaluation'  # for evaluation results
WALKS_DIR = OUT_DIR / 'walks'  # for storing ground truth walks
TABLES_DIR = OUT_DIR / 'tables'  # for storing tables, e.g. evaluation results
ATTENTION_DIR = OUT_DIR/ 'attention'  # for attention analysis results

# Cache
EVENTLOG_CACHE_DIR = CACHE_DIR / 'eventlogs'  # For caching datasets so the event log does not always have to be loaded
RESULT_DIR = CACHE_DIR / 'results'  # For caching anomaly detection results
CONFNET_RESULT_DIR = CACHE_DIR / 'confnet'  # For caching ConfNet alignments
ALIGNMENTS_DIR = CACHE_DIR / 'alignments'  # For caching optimal alignments
CORRECTIONS_DIR = CACHE_DIR / 'corrections'  # For caching the original features before applying anomalies

# Config
CONFIG_DIR = ROOT_DIR / '.config'

# Database
DATABASE_FILE = OUT_DIR / 'april.db'

# Extensions
MODEL_EXT = '.h5'

# Misc
DATE_FORMAT = 'YYYYMMDD-HHmmss.SSSSSS'


def generate():
    """Generate directories."""
    dirs = [
        ROOT_DIR,
        OUT_DIR,
        RES_DIR,
        CACHE_DIR,
        RESULT_DIR,
        EVENTLOG_CACHE_DIR,
        MODEL_DIR,
        CONFNET_DIR,
        CONFNET_RESULT_DIR,
        ALIGNMENTS_DIR,
        CORRECTIONS_DIR,
        PROCESS_MODEL_DIR,
        EVALUATION_DIR,
        WALKS_DIR,
        TABLES_DIR,
        ATTENTION_DIR,
        EVENTLOG_DIR,
        BPIC_DIR,
        PLOT_DIR,
        TENSORBOARD_DIR
    ]
    for d in dirs:
        if not d.exists():
            d.mkdir()


def split_eventlog_name(name):
    s = name.split('-')
    model = None
    p = None
    id = None
    if len(s) > 0:
        model = s[0]
    if len(s) > 1:
        p = float(s[1])
    if len(s) > 2:
        id = int(s[2])
    return model, p, id


def split_model_name(name):
    s = name.split('_')
    event_log_name = None
    ad = None
    date = None
    if len(s) > 0:
        event_log_name = s[0]
    if len(s) > 1:
        ad = s[1]
    if len(s) > 2:
        sd = s[2]
        if '.model' in sd:
            sd = sd[:sd.index('.model')]
        elif '.h5' in sd:
            sd = sd[:sd.index('.h5')]

        date = arrow.get(sd, DATE_FORMAT)
    return event_log_name, ad, date


class File(object):
    ext = None

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.file = self.path.name
        self.name = self.path.stem
        self.str_path = str(path)

    def remove(self):
        import os
        if self.path.exists():
            os.remove(self.path)


class EventLogFile(File):
    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        if '.json' not in path.suffixes:
            path = Path(str(path) + '.json.gz')
        if not path.is_absolute():
            path = EVENTLOG_DIR / path.name

        super(EventLogFile, self).__init__(path)

        if len(self.path.suffixes) > 1:
            self.name = Path(self.path.stem).stem

        self.model, self.p, self.id = split_eventlog_name(self.name)

    @property
    def cache_file(self):
        return EVENTLOG_CACHE_DIR / (self.name + '.h5')


class ModelFile(File):
    ext = MODEL_EXT

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        if path.suffix != self.ext:
            path = Path(str(path) + self.ext)
        if not path.is_absolute():
            path = MODEL_DIR / path.name

        super(ModelFile, self).__init__(path)

        self.event_log_name, self.model_name, self.date = split_model_name(self.name)
        self.model, self.p, self.id = split_eventlog_name(self.event_log_name)

    @property
    def result_file(self):
        return RESULT_DIR / (self.name + MODEL_EXT)


class AlignerFile(ModelFile):
    ext = MODEL_EXT

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        if path.suffix != self.ext:
            path = Path(str(path) + self.ext)
        if not path.is_absolute():
            path = CONFNET_DIR / path.name

        super(AlignerFile, self).__init__(path)

        self.event_log_name, self.ad, self.date = split_model_name(self.name)
        self.model, self.p, self.id = split_eventlog_name(self.event_log_name)

        self.use_case_attributes = None
        self.use_event_attributes = None
        if 'confnet' in self.ad:
            ea, ca = int(self.ad[-2:-1]), int(self.ad[-1:])
            self.use_case_attributes = bool(ca)
            self.use_event_attributes = bool(ea)
            # self.ad = self.ad[:-2]

    @property
    def result_file(self):
        return CONFNET_RESULT_DIR / (self.name + MODEL_EXT)


class ResultFile(File):
    ext = MODEL_EXT

    @property
    def model_file(self):
        return MODEL_DIR / (self.name + MODEL_EXT)


def get_event_log_files(path=None):
    if path is None:
        path = EVENTLOG_DIR
    for f in path.glob('*.json*'):
        yield EventLogFile(f)


def get_model_files(path=None):
    if path is None:
        path = MODEL_DIR
    for f in path.glob(f'*{MODEL_EXT}'):
        yield ModelFile(f)


def get_aligner_files(path=None):
    if path is None:
        path = CONFNET_DIR
    for f in path.glob(f'*{MODEL_EXT}'):
        yield AlignerFile(f)


def get_result_files(path=None):
    if path is None:
        path = RESULT_DIR
    for f in path.glob(f'*{MODEL_EXT}'):
        yield ResultFile(f)


def get_process_model_files(path=None):
    if path is None:
        path = PROCESS_MODEL_DIR
    for f in path.glob('*.plg'):
        yield f.stem


def download_bpic_logs():
    import gzip
    import requests
    from tqdm import tqdm

    logs = [
        ('BPIC12.xes.gz', 'https://data.4tu.nl/repository/uuid:3926db30-f712-4394-aebc-75976070e91f/DATA1'),
        ('BPIC13_closed_problems.xes.gz',
         'https://data.4tu.nl/repository/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11/DATA1'),
        ('BPIC13_incidents.xes.gz', 'https://data.4tu.nl/repository/uuid:500573e6-accc-4b0c-9576-aa5468b10cee/DATA1'),
        ('BPIC13_open_problems.xes.gz',
         'https://data.4tu.nl/repository/uuid:3537c19d-6c64-4b1d-815d-915ab0e479da/DATA1'),
        ('BPIC15_1.xes', 'https://data.4tu.nl/repository/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17/DATA1'),
        ('BPIC15_2.xes', 'https://data.4tu.nl/repository/uuid:63a8435a-077d-4ece-97cd-2c76d394d99c/DATA1'),
        ('BPIC15_3.xes', 'https://data.4tu.nl/repository/uuid:ed445cdd-27d5-4d77-a1f7-59fe7360cfbe/DATA1'),
        ('BPIC15_4.xes', 'https://data.4tu.nl/repository/uuid:679b11cf-47cd-459e-a6de-9ca614e25985/DATA1'),
        ('BPIC15_5.xes', 'https://data.4tu.nl/repository/uuid:b32c6fe5-f212-4286-9774-58dd53511cf8/DATA1'),
        ('BPIC17.xes.gz', 'https://data.4tu.nl/repository/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b/DATA1'),
        ('BPIC17_offer_log.xes.gz', 'https://data.4tu.nl/repository/uuid:7e326e7e-8b93-4701-8860-71213edf0fbe/DATA1'),
        ('BPIC18.xes.gz', 'https://data.4tu.nl/repository/uuid:3301445f-95e8-4ff0-98a4-901f1f204972/DATA1'),
        ('BPIC19.xes', 'https://data.4tu.nl/repository/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1/DATA'),
        ('hospital.xes.gz',
         'https://data.4tu.nl/repository/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741/DATA1'),
        ('sepsis.xes.gz',
         'https://data.4tu.nl/repository/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460/DATA1'),
        ('Road_Traffic_Fine_Management_Process.xes.gz',
         'https://data.4tu.nl/repository/uuid:270fd440-1057-4fb9-89a9-b699b47990f5/DATA1')
    ]

    for file_name, url in tqdm(logs):
        r = requests.get(url, allow_redirects=True)
        if file_name.endswith('.gz'):
            open(str(BPIC_DIR / file_name), 'wb').write(r.content)
        else:
            gzip.open(str(BPIC_DIR / file_name) + '.gz', 'wb').write(r.content)
