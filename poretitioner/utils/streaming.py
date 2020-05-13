import grpc
import minknow.rpc.data_pb2 as data
import minknow.rpc.data_pb2_grpc as data_rpc
import minknow.rpc.manager_pb2 as manager
import minknow.rpc.manager_pb2_grpc as manager_grpc
import minknow.rpc.protocol_pb2 as protocol


class MinKnow:

    MINKNOW_SERVER = "localhost"
    MINKNOW_PORT = "9501"  # Pre-defined

    def __init__(self, server=MINKNOW_SERVER, port=MINKNOW_PORT):
        address = f"{server}:{port}"
        self.channel = grpc.insecure_channel(address)

    def get_data(self):
        datarfpc = data_rpc.DataServiceStub(self.channel)
        datarpc.get_data_types(data.GetDataTypesRequest())

        # TODO: Fill in
