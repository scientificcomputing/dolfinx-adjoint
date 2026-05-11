import dolfinx
import numpy as np
from pyadjoint.overloaded_type import FloatingType, register_overloaded_type, create_overloaded_object
from pyadjoint.tape import no_annotations


class Constant(dolfinx.fem.Constant, FloatingType):
    def __init__(self, domain, c, **kwargs):
        super().__init__(domain, c)
        FloatingType.__init__(
            self,
            domain,
            c,
            block_class=kwargs.pop("block_class", None),
            _ad_floating_active=kwargs.pop("_ad_floating_active", False),
            _ad_args=kwargs.pop("_ad_args", None),
            output_block_class=kwargs.pop("output_block_class", None),
            _ad_output_args=kwargs.pop("_ad_output_args", None),
            _ad_outputs=kwargs.pop("_ad_outputs", None),
            annotate=kwargs.pop("annotate", True),
            **kwargs,
        )

    def _ad_init_object(self, obj):
        return type(self)(self.ufl_domain(), obj)

    @no_annotations
    def _ad_create_checkpoint(self):
        # Create a checkpoint using the underlying value
        return create_overloaded_object(dolfinx.fem.Constant(None, self.value.copy()))

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_dot(self, other, options=None):
        return np.sum(self.value * other.value)

    def _ad_copy(self):
        return Constant(self.ufl_domain(), self.value.copy())


register_overloaded_type(Constant, (dolfinx.fem.Constant, Constant))
