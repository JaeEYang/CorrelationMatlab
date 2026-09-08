from correlation2d3d.session import CorrelationSession


class OfflineCorrelationController:

    def __init__(
        self,
        viewer,
        session: CorrelationSession,
    ):
        self.viewer = viewer
        self.session = session

    # this becomes out generic loader for modalities make it bit tidy to keep track of session state.
    # retuns the correspoding state object
    def _get_modality(self, role: str):
        if role == "FLM":
            return self.session.flm

        if role == "TEM":
            return self.session.tem

        raise ValueError(
            f"unknown modality role: {role}"
        )

    # helper if layer exists we can remove it if not do nothing.
    def _remove_layer_if_present(
        self,
        layer_name: str,
    ) -> None:
        try:
            layer = self.viewer.layers[layer_name]
        except KeyError:
            return

        self.viewer.layers.remove(layer)