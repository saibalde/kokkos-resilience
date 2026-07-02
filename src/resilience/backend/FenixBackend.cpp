/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */
#include "FenixBackend.hpp"

#include <sstream>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>

#include <fenix.h>

namespace KokkosResilience {

namespace detail {

void report_error(const std::string& msg) { Kokkos::Impl::throw_runtime_exception(msg); }

Registration unalias_member(const std::unordered_map<std::string, Registration>& alias_map,
                            const Registration& member) {
  auto alias_iter = alias_map.find(member->name);
  if (alias_iter != alias_map.end()) {
    return unalias_member(alias_map, alias_iter->second);
  }
  return member;
}

std::unordered_set<Registration> get_unaliased_member_list(
    const std::unordered_map<std::string, Registration>& alias_map, const std::unordered_set<Registration>& members) {
  std::unordered_set<Registration> unaliased_members;
  for (auto&& member : members) {
    unaliased_members.insert(unalias_member(alias_map, member));
  }
  return unaliased_members;
}

void create_data_group(MPI_Comm mpi_comm, int group_id, const std::string& group_label) {
  int mpi_size;
  MPI_Comm_size(mpi_comm, &mpi_size);

  const int start_time_stamp = 0;
  const int checkpoint_depth = 0;
  const int policy_name      = FENIX_DATA_POLICY_IN_MEMORY_RAID;
  int policy_value[3]        = {1, std::max(1, mpi_size / 2), 0};

  int flag;
  int status =
      Fenix_Data_group_create(group_id, mpi_comm, start_time_stamp, checkpoint_depth, policy_name, policy_value, &flag);
  if (status != FENIX_SUCCESS) {
    std::ostringstream msg;
    msg << "failed to create data group \"" << group_label << "\": fenix return code = " << status
        << ", fenix error flag = " << flag;
    report_error(msg.str());
  }
}

void commit_data_group(int group_id, const std::string& group_label) {
  int time_stamp;
  int status = Fenix_Data_commit(group_id, &time_stamp);
  if (status != FENIX_SUCCESS) {
    std::ostringstream msg;
    msg << "failed to commit data group \"" << group_label << "\"; fenix return code = " << status;
    report_error(msg.str());
  }
}

void create_data_member(int group_id, const std::string& group_label, int member_id, const std::string& member_label,
                        void* data, int count) {
  int status = Fenix_Data_member_create(group_id, member_id, data, count, MPI_CHAR);
  if (status != FENIX_SUCCESS) {
    std::ostringstream msg;
    msg << "failed to create data member \"" << member_label << "\" in data group \"" << group_label
        << "\"; fenix return code = " << status;
    report_error(msg.str());
  }
}

void update_data_member(int group_id, const std::string& group_label, int member_id, const std::string& member_label,
                        void* data, int count) {
  {
    int flag;
    int status = Fenix_Data_member_attr_set(group_id, member_id, FENIX_DATA_MEMBER_ATTRIBUTE_BUFFER, data, &flag);
    if (status != FENIX_SUCCESS) {
      std::ostringstream msg;
      msg << "failed to set buffer attribute of data member \"" << member_label << "\" in data group \"" << group_label
          << "\"; fenix return code = " << status << ", fenix error flag = " << flag;
      report_error(msg.str());
    }
  }
  {
    int flag;
    int status = Fenix_Data_member_attr_set(group_id, member_id, FENIX_DATA_MEMBER_ATTRIBUTE_COUNT, &count, &flag);
    if (status != FENIX_SUCCESS) {
      std::ostringstream msg;
      msg << "failed to set count attribute of data member \"" << member_label << "\" in data group \"" << group_label
          << "\"; fenix return code = " << status << ", fenix error flag = " << flag;
      report_error(msg.str());
    }
  }
}

void store_data_member(int group_id, const std::string& group_label, int member_id, const std::string& member_label) {
  int status = Fenix_Data_member_store(group_id, member_id, FENIX_DATA_SUBSET_FULL);
  if (status != FENIX_SUCCESS) {
    std::ostringstream msg;
    msg << "failed to store data member \"" << member_label << "\" in data group \"" << group_label
        << "; fenix return code = " << status;
    report_error(msg.str());
  }
}

void restore_data_member(int group_id, const std::string& group_label, int member_id, const std::string& member_label,
                         int time_stamp, void* data, int count) {
  int status = Fenix_Data_member_restore(group_id, member_id, data, count, time_stamp, NULL);
  if (status != FENIX_SUCCESS) {
    std::ostringstream msg;
    msg << "failed to restore data member \"" << member_label << "\" in data group \"" << group_label
        << "\"; fenix return code = " << status;
    report_error(msg.str());
  }
}

void assert_data_member_exists(const std::unordered_map<int, std::unordered_set<int>>& group_members, int group_id,
                               const std::string& group_label, int member_id, const std::string& member_label) {
  const auto& member_ids = group_members.find(group_id)->second;
  if (member_ids.find(member_id) == member_ids.end()) {
    std::ostringstream msg;
    msg << "data member \"" << member_label << "\" in data group \"" << group_label << "\" does not exist";
    report_error(msg.str());
  }
}

int get_time_stamp_of_snapshot_at_position(int group_id, const std::string& group_label, int position) {
  int time_stamp;
  int status = Fenix_Data_group_get_snapshot_at_position(group_id, position, &time_stamp);
  if (status != FENIX_SUCCESS) {
    std::ostringstream msg;
    msg << "failed to retrieve time stamp of snapshot at position " << position << " in data group \"" << group_label
        << "\"";
    report_error(msg.str());
  }
  return time_stamp;
}

}  // namespace detail

FenixMemoryBackend::FenixMemoryBackend(ContextBase& ctx, MPI_Comm mpi_comm) : m_context(&ctx), m_mpi_comm(mpi_comm) {}

FenixMemoryBackend::~FenixMemoryBackend() {
  clear_checkpoints();
  reset();
}

void FenixMemoryBackend::checkpoint(const std::string& label, int version,
                                    const std::unordered_set<Registration>& members) {
  const int group_id = static_cast<int>(label_hash(label));

  if (m_group_members.find(group_id) == m_group_members.end()) {
    detail::create_data_group(m_mpi_comm, group_id, label);
    m_group_members.emplace(group_id, std::unordered_set<int>());
  }

  auto& members_list = m_group_members.find(group_id)->second;

  // store version information in the checkpoint
  if (members_list.find(member_id_of_version) == members_list.end()) {
    detail::create_data_member(group_id, label, member_id_of_version, "version", &version, sizeof(int));
    members_list.emplace(member_id_of_version);
  } else {
    detail::update_data_member(group_id, label, member_id_of_version, "version", &version, sizeof(int));
  }

  detail::store_data_member(group_id, label, member_id_of_version, "version");

  auto unaliased_members = detail::get_unaliased_member_list(m_alias_map, members);

  // store actual members alongside their size information
  for (auto&& member : unaliased_members) {
    std::vector<char> buffer;
    auto sink = boost::iostreams::back_inserter(buffer);
    boost::iostreams::stream<decltype(sink)> stream(sink);

    member->serialize(stream);
    stream.flush();

    char* data = buffer.data();
    int count  = buffer.size();

    const int member_hash = static_cast<int>(member->hash());

    const int count_id  = member_id_offset + 2 * member_hash;
    const int member_id = count_id + 1;

    if (members_list.find(count_id) == members_list.end()) {
      detail::create_data_member(group_id, label, count_id, member->name + " count", &count, sizeof(int));
      members_list.emplace(count_id);
    } else {
      detail::update_data_member(group_id, label, count_id, member->name + " count", &count, sizeof(int));
    }

    if (members_list.find(member_id) == members_list.end()) {
      detail::create_data_member(group_id, label, member_id, member->name, data, count);
      members_list.emplace(member_id);
    } else {
      detail::update_data_member(group_id, label, member_id, member->name, data, count);
    }

    detail::store_data_member(group_id, label, count_id, member->name + " count");
    detail::store_data_member(group_id, label, member_id, member->name);
  }

  detail::commit_data_group(group_id, label);

  m_latest_version[label] = version;
}

void FenixMemoryBackend::restart(const std::string& label, int version, std::unordered_set<Registration>& members) {
  const int group_id = label_hash(label);

  if (m_group_members.find(group_id) == m_group_members.end()) {
    std::ostringstream msg;
    msg << "data group \"" << label << "\" does not exist";
    detail::report_error(msg.str());
  }

  detail::assert_data_member_exists(m_group_members, group_id, label, member_id_of_version, "version");
  int time_stamp;
  {
    int num_snapshot;
    {
      int status = Fenix_Data_group_get_number_of_snapshots(group_id, &num_snapshot);
      if (status != FENIX_SUCCESS) {
        std::ostringstream msg;
        msg << "failed to retrieve number of snapshots in data group \"" << label
            << "\"; fenix return code = " << status;
        detail::report_error(msg.str());
      }
    }

    if (num_snapshot == 0) {
      std::ostringstream msg;
      msg << "there are no snapshots in data group \"" << label << "\"";
      detail::report_error(msg.str());
    }

    int position = 0;
    while (position < num_snapshot) {
      time_stamp = detail::get_time_stamp_of_snapshot_at_position(group_id, label, position);

      int current_version;
      detail::restore_data_member(group_id, label, member_id_of_version, "version", time_stamp, &current_version,
                                  sizeof(int));

      if (current_version == version) {
        break;
      }

      ++position;
    }

    if (position == num_snapshot) {
      std::ostringstream msg;
      msg << "could not find version " << version << " of snapshot in data group \"" << label;
      detail::report_error(msg.str());
    }
  }

  auto unaliased_members = detail::get_unaliased_member_list(m_alias_map, members);

  for (auto&& member : unaliased_members) {
    const int member_hash = static_cast<int>(member->hash());

    const int count_id  = member_id_offset + 2 * member_hash;
    const int member_id = count_id + 1;

    detail::assert_data_member_exists(m_group_members, group_id, label, count_id, member->name + " count");
    detail::assert_data_member_exists(m_group_members, group_id, label, member_id, member->name);

    int count;
    detail::restore_data_member(group_id, label, count_id, member->name + " count", time_stamp, &count, sizeof(int));

    std::vector<char> buffer(count);
    detail::restore_data_member(group_id, label, member_id, member->name, time_stamp, buffer.data(), count);

    boost::iostreams::array_source source(buffer.data(), buffer.size());
    boost::iostreams::stream<decltype(source)> stream(source);

    member->deserialize(stream);
  }
}

int FenixMemoryBackend::latest_version(const std::string& label) const noexcept {
  auto iter = m_latest_version.find(label);
  if (iter != m_latest_version.end()) {
    return iter->second;
  }

  const int group_id = label_hash(label);

  auto group_members_iter = m_group_members.find(group_id);
  if (group_members_iter == m_group_members.end()) {
    return -1;
  }

  auto& group_members = group_members_iter->second;
  if (group_members.find(member_id_of_version) == group_members.end()) {
    return -1;
  }

  const int position   = 0;  // latest snapshot is at position 0
  const int time_stamp = detail::get_time_stamp_of_snapshot_at_position(group_id, label, position);

  int version;
  detail::restore_data_member(group_id, label, member_id_of_version, "version", time_stamp, &version, sizeof(int));

  m_latest_version[label] = version;

  return version;
}

bool FenixMemoryBackend::restart_available(const std::string& label, int version) {
  return version == latest_version(label);
}

void FenixMemoryBackend::clear_checkpoints() {
  for (auto& [group_id, group_members] : m_group_members) {
    for (auto& member_id : group_members) {
      Fenix_Data_member_delete(group_id, member_id);
    }
    group_members.clear();
  }
  // not deleteting data groups to avoid double free in fenix
}

void FenixMemoryBackend::reset() {
  m_latest_version.clear();
  m_alias_map.clear();
}

void FenixMemoryBackend::register_alias(Registration& member, const std::string& alias) {
  m_alias_map.try_emplace(alias, member);
}

}  // namespace KokkosResilience
