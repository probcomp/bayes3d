# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def static_check_supports(target, proposal):
    proposal_trace_type = proposal.get_trace_type(target)
    target_trace_type = target.get_trace_type()
    check, mismatch = target_trace_type.on_support(proposal_trace_type)
    if not check:
        raise Exception(
            f"Trace type mismatch.\n"
            f"Given target: {target}\n"
            f"Proposal: {proposal}\n"
            f"\nAbsolute continuity failure at the following addresses:\n{mismatch}"
        )
