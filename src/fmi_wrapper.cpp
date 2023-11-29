#include <fmi.h>

void fmi_send(void* buf, std::size_t bytes, unsigned int dest, FMI::Communicator& comm) {
    comm.send_raw(buf, bytes, dest);
}

void fmi_recv(void* buf, std::size_t bytes, unsigned int src, FMI::Communicator& comm) {
    comm.recv_raw(buf, bytes, src);
}


void fmi_scatter(void* send_buf, std::size_t send_bytes, void* recv_buf, std::size_t recv_bytes, unsigned int root, FMI::Communicator& comm) {
    comm.scatter_raw(send_buf, send_bytes, recv_buf, recv_bytes, root);
}