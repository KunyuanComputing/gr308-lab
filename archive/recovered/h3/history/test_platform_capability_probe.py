import unittest
from platform_capability_probe import finalize


class DiagnosticBoundaryTests(unittest.TestCase):
    def test_operator_success_does_not_imply_platform_or_e2e(self):
        result = finalize({'platform_device_type': 'cpu',
                           'operators': [{'passed': True}] * 6})
        self.assertTrue(result['small_ops_passed'])
        self.assertFalse(result['platform_is_fant'])
        self.assertFalse(result['model_e2e'])
        self.assertFalse(result['native_kernel_acceleration_proven'])

    def test_partial_tests_are_not_success(self):
        result = finalize({'platform_device_type': 'fant',
                           'operators': [{'passed': True}] * 5})
        self.assertTrue(result['platform_is_fant'])
        self.assertFalse(result['small_ops_passed'])

    def test_one_failure_rejects_operator_gate(self):
        result = finalize({'operators': [{'passed': True}] * 5 + [{'passed': False}]})
        self.assertFalse(result['small_ops_passed'])


if __name__ == '__main__':
    unittest.main()
