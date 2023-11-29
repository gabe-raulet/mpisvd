#pragma once

#include <fmi.h>

void fmi_send(void* buf, std::size_t bytes, unsigned int dest, FMI::Communicator& comm);
void fmi_recv(void* buf, std::size_t bytes, unsigned int src, FMI::Communicator& comm);
void fmi_scatter(void* send_buf, std::size_t send_bytes, void* recv_buf, std::size_t recv_bytes, unsigned int root, FMI::Communicator& comm);