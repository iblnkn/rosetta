# Copyright 2025 Isaac Blankenau
#
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

"""The header timeline is attested by field type, not field name."""

from rosetta_ros.timelines import TIMELINES


def test_header_timeline_attests_field_type_not_name():
    # A field merely *named* header must not validate a timeline the
    # extractor can never produce. No stock message has a non-Header `header`
    # field, so this goes through the registry with stand-in classes.
    class WrongHeader:
        @staticmethod
        def get_fields_and_field_types():
            return {"header": "uint32"}

    class RealHeader:
        @staticmethod
        def get_fields_and_field_types():
            return {"header": "std_msgs/Header"}

    assert not TIMELINES["header"].provides(WrongHeader)
    assert TIMELINES["header"].provides(RealHeader)
