import logging
from typing import Iterable, NewType, Optional

import grpc
import minknow.rpc.acquisition_pb2 as acquisition
import minknow.rpc.acquisition_pb2_grpc as acquisition_grpc
import minknow.rpc.analysis_configuration_pb2 as analysis_config
import minknow.rpc.analysis_configuration_pb2_grpc as analysis_config_rpc
import minknow.rpc.data_pb2 as data
import minknow.rpc.data_pb2_grpc as data_rpc
import minknow.rpc.manager_pb2 as manager
import minknow.rpc.manager_pb2_grpc as manager_rpc
import minknow.rpc.protocol_pb2 as protocol
from google.protobuf.wrappers_pb2 import StringValue

ActiveDevice = NewType("ActiveDevice", manager.ListDevicesResponse.ActiveDevice)


def get_address(server: str, port: str):
    address = f"{server}:{port}"
    return address


class MinknowDevice:
    def __init__(self, device: ActiveDevice, server="localhost", logger=None):
        self.device = device
        port = device.ports.insecure_grpc
        address = get_address(server, port)
        self.channel = grpc.insecure_channel(address)

        self.logger = logger or logging.getLogger(__name__)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.channel.close()

    def start_experiment(self, group_id="Expiriment00"):
        protocol_user_info = protocol.ProtocolRunUserInfo(
            protocol_group_id=StringValue(value=group_id)
        )


class MinKnow:
    def __init__(self, server="localhost", port=9501, logger=None):
        """An adapter class for interacting with Oxford Nanopore Technology nanopore devices.
        This class provides a subset of MinKNOW functionality, and can be used
        for running protocols remotely and streaming data directly from devices like the minION.

        This class also abstracts away the serialization protocol details, providing a
        high-level interface.

        Parameters
        ----------
        server : str, optional
            The hostname of the server hosting the nanopore device, by default "localhost"
        port : int, optional
            The server port that MinKNOW should connect to, by default 9501
        port : logger, optional
            The logger to use for logging. Uses default namespaced logger by default if a logger is not provided.
        """
        address = f"{server}:{port}"
        # TODO: Make this a context manager to automatically close connection when finished.
        self.channel = grpc.insecure_channel(address)

        self.logger = logger or logging.getLogger(__name__)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.channel.close()

    def list_devices(self) -> Iterable[ActiveDevice]:
        """Fetches a list of active nanopore devices.

        Returns
        -------
        Iterable[ActiveDevice]
            An iterable of active devices. Choose one of these nanopore devices to act on in subsequent steps.
        """
        stub = manager_rpc.ManagerServiceStub(self.channel)
        request = manager.ListDevicesRequest()
        response = stub.list_devices(request)
        devices = response.active
        return devices

    def device_by_name(self, name: str) -> ActiveDevice:
        """Gets the nanopore device by its name.
        Name usually takes the form (expressed here in regex) "[A-Z]{2}[0-9]{5}", e.g. MN27856

        Parameters
        ----------
        name : str
            Name of the device.
            Oxford Nanopore devices usually have names like "MN27856", where the
            digits section can be found printed on the physical device itself.

        Returns
        -------
        ActiveDevice
            The device with the given name.

        Raises
        ------
        KeyError
            If no device of that name can be found. Try plugging the device back in.
        """
        for device in self.list_devices():
            if device.name == name:
                return device

        message = f"No device named '{name}' found."
        self.logger.exception(message)
        raise KeyError(message)

    # def configure_acquisition(self, device: ActiveDevice):
    #     analysis_config.AnalysisConfiguration()
    #     stub = analysis_config_rpc.AnalysisConfigurationServiceStub(self.channel)

    # def get_data(self):
    #     datarfpc = data_rpc.DataServiceStub(self.channel)
    #     datarpc.get_data_types(data.GetDataTypesRequest())

    # TODO: Fill in
