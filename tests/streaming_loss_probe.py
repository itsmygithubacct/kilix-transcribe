"""Retained owned-loss observer applied to the incremental worker pipe."""
import owned_loss_probe
import test_leased_runtime

original = test_leased_runtime.value


def streaming_value(*args, **kwargs):
    value = original(*args, **kwargs)
    value['args']['stream'] = 'segments-v1'
    return value


test_leased_runtime.value = streaming_value
if __name__ == '__main__':
    owned_loss_probe.main()
