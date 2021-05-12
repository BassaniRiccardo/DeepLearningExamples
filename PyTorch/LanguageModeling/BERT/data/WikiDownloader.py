# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bz2
import os
import urllib.request
import sys
import subprocess

class WikiDownloader:
    def __init__(self, language, save_path):
        self.save_path = save_path + '/wikicorpus_' + language

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.language = language
        self.download_urls = {
            'ar' : 'https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2',
            'bn' : 'https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-articles.xml.bz2',
            'en' : 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2',
            'fi' : 'https://dumps.wikimedia.org/fiwiki/latest/fiwiki-latest-pages-articles.xml.bz2',
            'id' : 'https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2',
            'ko' : 'https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2',
            'ru' : 'https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2',
            'sw' : 'https://dumps.wikimedia.org/swwiki/latest/swwiki-latest-pages-articles.xml.bz2',
            'te' : 'https://dumps.wikimedia.org/tewiki/latest/tewiki-latest-pages-articles.xml.bz2',
            'zh' : 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2'
        }

        self.output_files = {
            'ar' : 'wikicorpus_en.xml.bz2',
            'bn' : 'wikicorpus_bn.xml.bz2',
            'en' : 'wikicorpus_en.xml.bz2',
            'fi' : 'wikicorpus_fi.xml.bz2',
            'id' : 'wikicorpus_id.xml.bz2',
            'ko' : 'wikicorpus_ko.xml.bz2',
            'ru' : 'wikicorpus_ru.xml.bz2',
            'sw' : 'wikicorpus_sw.xml.bz2',
            'te' : 'wikicorpus_te.xml.bz2',
            'zh' : 'wikicorpus_zh.xml.bz2'
        }


    def download(self):
        if self.language in self.download_urls:
            url = self.download_urls[self.language]
            filename = self.output_files[self.language]

            print('Downloading:', url)
            if os.path.isfile(self.save_path + '/' + filename):
                print('** Download file already exists, skipping download')
            else:
                cmd = ['wget', url, '--output-document={}'.format(self.save_path + '/' + filename)]
                print('Running:', cmd)
                status = subprocess.run(cmd)
                if status.returncode != 0:
                    raise RuntimeError('Wiki download not successful')

            # Always unzipping since this is relatively fast and will overwrite
            print('Unzipping:', self.output_files[self.language])
            subprocess.run('bzip2 -dk ' + self.save_path + '/' + filename, shell=True, check=True)

        else:
            assert False, 'WikiDownloader not implemented for this language yet.'

