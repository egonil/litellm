import json
import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))
from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider as OTLoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


class TestOpenTelemetryGuardrails(unittest.TestCase):
    @patch("litellm.integrations.opentelemetry.datetime")
    def test_create_guardrail_span_with_valid_info(self, mock_datetime):
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()
        mock_span = MagicMock()
        otel.tracer.start_span.return_value = mock_span

        # Create guardrail information
        guardrail_info = {
            "guardrail_name": "test_guardrail",
            "guardrail_mode": "input",
            "masked_entity_count": {"CREDIT_CARD": 2},
            "guardrail_response": "filtered_content",
            "start_time": 1609459200.0,
            "end_time": 1609459201.0,
        }

        # Create a kwargs dict with standard_logging_object containing guardrail information
        kwargs = {
            "standard_logging_object": {"guardrail_information": [guardrail_info]}
        }

        # Call the method
        otel._create_guardrail_span(kwargs=kwargs, context=None)

        # Assertions
        otel.tracer.start_span.assert_called_once()

        # print all calls to mock_span.set_attribute
        print("Calls to mock_span.set_attribute:")
        for call in mock_span.set_attribute.call_args_list:
            print(call)

        # Check that the span has the correct attributes set
        mock_span.set_attribute.assert_any_call("guardrail_name", "test_guardrail")
        mock_span.set_attribute.assert_any_call("guardrail_mode", "input")
        mock_span.set_attribute.assert_any_call(
            "guardrail_response", "filtered_content"
        )
        mock_span.set_attribute.assert_any_call(
            "masked_entity_count", safe_dumps({"CREDIT_CARD": 2})
        )

        # Verify that the span was ended
        mock_span.end.assert_called_once()

    def test_create_guardrail_span_with_no_info(self):
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()

        # Test with no guardrail information
        kwargs = {"standard_logging_object": {}}
        otel._create_guardrail_span(kwargs=kwargs, context=None)

        # Verify that start_span was never called
        otel.tracer.start_span.assert_not_called()


class TestOpenTelemetryCostBreakdown(unittest.TestCase):
    def test_cost_breakdown_emitted_to_otel_span(self):
        """
        Test that cost breakdown from StandardLoggingPayload is emitted to OpenTelemetry span attributes.
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        cost_breakdown = {
            "input_cost": 0.001,
            "output_cost": 0.002,
            "total_cost": 0.003,
            "tool_usage_cost": 0.0001,
            "original_cost": 0.004,
            "discount_percent": 0.25,
            "discount_amount": 0.001,
        }

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
                "cost_breakdown": cost_breakdown,
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        mock_span.set_attribute.assert_any_call("gen_ai.cost.input_cost", 0.001)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.output_cost", 0.002)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.total_cost", 0.003)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.tool_usage_cost", 0.0001)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.original_cost", 0.004)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.discount_percent", 0.25)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.discount_amount", 0.001)

    def test_cost_breakdown_with_partial_fields(self):
        """
        Test that cost breakdown works correctly when only some fields are present.
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        cost_breakdown = {
            "input_cost": 0.001,
            "output_cost": 0.002,
            "total_cost": 0.003,
        }

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
                "cost_breakdown": cost_breakdown,
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        mock_span.set_attribute.assert_any_call("gen_ai.cost.input_cost", 0.001)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.output_cost", 0.002)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.total_cost", 0.003)

        call_args_list = [call[0] for call in mock_span.set_attribute.call_args_list]
        assert ("gen_ai.cost.tool_usage_cost", 0.0001) not in call_args_list
        assert ("gen_ai.cost.original_cost", 0.004) not in call_args_list


class TestOpenTelemetryProviderInitialization(unittest.TestCase):
    """Test suite for verifying provider initialization respects existing providers"""

    def test_init_tracing_respects_existing_tracer_provider(self):
        """
        Unit test: _init_tracing() should respect existing TracerProvider.

        When a TracerProvider already exists (e.g., set by Langfuse SDK),
        LiteLLM should use it instead of creating a new one.
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        # Setup: Create and set an existing TracerProvider
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        existing_provider = trace.get_tracer_provider()

        # Act: Initialize OpenTelemetry integration (should detect existing provider)
        otel_integration = OpenTelemetry()

        # Assert: The existing provider should still be active
        current_provider = trace.get_tracer_provider()
        assert (
            current_provider is existing_provider
        ), "Existing TracerProvider should be respected and not overridden"

    @patch.dict(
        os.environ, {"LITELLM_OTEL_INTEGRATION_ENABLE_METRICS": "true"}, clear=True
    )
    def test_init_metrics_respects_existing_meter_provider(self):
        """
        Unit test: _init_metrics() should respect existing MeterProvider.

        When a MeterProvider already exists (e.g., set by Langfuse SDK),
        LiteLLM should use it instead of creating a new one.
        """
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider

        # Create and set an existing MeterProvider
        meter_provider = MeterProvider()
        metrics.set_meter_provider(meter_provider)
        existing_provider = metrics.get_meter_provider()

        # Act: Initialize OpenTelemetry integration (should detect existing provider)
        config = OpenTelemetryConfig.from_env()
        otel_integration = OpenTelemetry(config=config)

        # Assert: The existing provider should still be active
        current_provider = metrics.get_meter_provider()
        assert (
            current_provider is existing_provider
        ), "Existing MeterProvider should be respected and not overridden"

    @patch.dict(
        os.environ, {"LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS": "true"}, clear=True
    )
    def test_init_logs_respects_existing_logger_provider(self):
        """
        Unit test: _init_logs() should respect existing LoggerProvider.

        When a LoggerProvider already exists (e.g., set by Langfuse SDK),
        LiteLLM should use it instead of creating a new one.
        """
        from opentelemetry._logs import get_logger_provider, set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider as OTLoggerProvider

        # Create and set an existing LoggerProvider
        logger_provider = OTLoggerProvider()
        set_logger_provider(logger_provider)
        existing_provider = get_logger_provider()

        # Act: Initialize OpenTelemetry integration (should detect existing provider)
        config = OpenTelemetryConfig.from_env()
        otel_integration = OpenTelemetry(config=config)

        # Assert: The existing provider should still be active
        current_provider = get_logger_provider()
        assert (
            current_provider is existing_provider
        ), "Existing LoggerProvider should be respected and not overridden"


class TestOpenTelemetry(unittest.TestCase):
    POLL_INTERVAL = 0.05
    POLL_TIMEOUT = 2.0
    MODEL = "arn:aws:bedrock:us-west-2:1234567890123:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    HERE = os.path.dirname(__file__)

    @patch.dict(os.environ, {}, clear=True)
    def test_open_telemetry_config_manual_defaults(self):
        """Manual OpenTelemetryConfig creation should populate default identifiers."""
        config = OpenTelemetryConfig(exporter="console", endpoint="http://collector")
        self.assertEqual(config.service_name, "litellm")
        self.assertEqual(config.deployment_environment, "production")
        self.assertEqual(config.model_id, "litellm")

    @patch.dict(os.environ, {}, clear=True)
    def test_open_telemetry_config_custom_service_name(self):
        """Model ID should inherit provided service name when not explicitly set."""
        config = OpenTelemetryConfig(service_name="custom-service", exporter="console")
        self.assertEqual(config.service_name, "custom-service")
        self.assertEqual(config.deployment_environment, "production")
        self.assertEqual(config.model_id, "custom-service")

    @patch.dict(os.environ, {}, clear=True)
    def test_open_telemetry_config_auto_infer_otlp_http_when_endpoint_set(self):
        """When endpoint is set but exporter is default 'console', auto-infer 'otlp_http'.

        This fixes an issue where UI-configured OTEL settings would default to console
        output instead of sending traces to the configured endpoint.
        See: https://github.com/BerriAI/litellm/issues/XXXX
        """
        # When endpoint is specified without explicit exporter, should auto-infer otlp_http
        config = OpenTelemetryConfig(endpoint="https://otel-collector.example.com:443")
        self.assertEqual(config.exporter, "otlp_http")

        # When exporter is explicitly set to something other than console, should not override
        config_grpc = OpenTelemetryConfig(
            exporter="grpc", endpoint="https://otel-collector.example.com:443"
        )
        self.assertEqual(config_grpc.exporter, "grpc")

        # When no endpoint is set, should keep console as default
        config_no_endpoint = OpenTelemetryConfig()
        self.assertEqual(config_no_endpoint.exporter, "console")

    def wait_for_spans(self, exporter: InMemorySpanExporter, prefix: str):
        """Poll until we see at least one span with an attribute key starting with `prefix`."""
        deadline = time.time() + self.POLL_TIMEOUT
        while time.time() < deadline:
            spans = exporter.get_finished_spans()
            matches = [
                s
                for s in spans
                if s.attributes and any(str(k).startswith(prefix) for k in s.attributes)
            ]
            if matches:
                return matches
            time.sleep(self.POLL_INTERVAL)
        return []

    def wait_for_metric(self, reader: InMemoryMetricReader, name: str):
        """Poll until we see a metric with the given name."""
        deadline = time.time() + self.POLL_TIMEOUT
        while time.time() < deadline:
            data = reader.get_metrics_data()
            # guard against None or missing attribute
            if not data or not hasattr(data, "resource_metrics"):
                time.sleep(self.POLL_INTERVAL)
                continue

            for rm in data.resource_metrics:
                for sm in rm.scope_metrics:
                    for m in sm.metrics:
                        if m.name == name:
                            return m

            time.sleep(self.POLL_INTERVAL)
        return None

    def wait_for_log(self, reader: InMemoryLogExporter, name: str):
        """Poll until we see a log with the given name."""
        deadline = time.time() + self.POLL_TIMEOUT
        while time.time() < deadline:
            logs = reader.get_finished_logs()
            if not logs:
                time.sleep(self.POLL_INTERVAL)
                continue
            matches = [
                log
                for log in logs
                # if log.attributes and any(str(k).startswith(prefix) for k in log.attributes)
            ]
            if matches:
                return matches
            time.sleep(self.POLL_INTERVAL)
        return []

    @patch("litellm.integrations.opentelemetry.datetime")
    def test_create_guardrail_span_with_valid_info(self, mock_datetime):
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()
        mock_span = MagicMock()
        otel.tracer.start_span.return_value = mock_span

        # Create guardrail information
        guardrail_info = {
            "guardrail_name": "test_guardrail",
            "guardrail_mode": "input",
            "masked_entity_count": {"CREDIT_CARD": 2},
            "guardrail_response": "filtered_content",
            "start_time": 1609459200.0,
            "end_time": 1609459201.0,
        }

        # Create a kwargs dict with standard_logging_object containing guardrail information
        kwargs = {
            "standard_logging_object": {"guardrail_information": [guardrail_info]}
        }

        # Call the method
        otel._create_guardrail_span(kwargs=kwargs, context=None)

        # Assertions
        otel.tracer.start_span.assert_called_once()

        # print all calls to mock_span.set_attribute
        print("Calls to mock_span.set_attribute:")
        for call in mock_span.set_attribute.call_args_list:
            print(call)

        # Check that the span has the correct attributes set
        mock_span.set_attribute.assert_any_call("guardrail_name", "test_guardrail")
        mock_span.set_attribute.assert_any_call("guardrail_mode", "input")
        mock_span.set_attribute.assert_any_call(
            "guardrail_response", "filtered_content"
        )
        mock_span.set_attribute.assert_any_call(
            "masked_entity_count", safe_dumps({"CREDIT_CARD": 2})
        )

        # Verify that the span was ended
        mock_span.end.assert_called_once()

    def test_create_guardrail_span_with_no_info(self):
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()

        # Test with no guardrail information
        kwargs = {"standard_logging_object": {}}
        otel._create_guardrail_span(kwargs=kwargs, context=None)

        # Verify that start_span was never called
        otel.tracer.start_span.assert_not_called()

    def test_get_tracer_to_use_for_request_with_dynamic_headers(self):
        """Test that get_tracer_to_use_for_request returns a dynamic tracer when dynamic headers are present."""
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()

        # Mock the dynamic header extraction and tracer creation
        with patch.object(
            otel, "_get_dynamic_otel_headers_from_kwargs"
        ) as mock_get_headers, patch.object(
            otel, "_get_tracer_with_dynamic_headers"
        ) as mock_get_tracer:
            # Test case 1: With dynamic headers
            mock_get_headers.return_value = {
                "arize-space-id": "test-space",
                "api_key": "test-key",
            }
            mock_dynamic_tracer = MagicMock()
            mock_get_tracer.return_value = mock_dynamic_tracer

            kwargs = {
                "standard_callback_dynamic_params": {"arize_space_key": "test-space"}
            }
            result = otel.get_tracer_to_use_for_request(kwargs)

            # Assertions
            mock_get_headers.assert_called_once_with(kwargs)
            mock_get_tracer.assert_called_once_with(
                {"arize-space-id": "test-space", "api_key": "test-key"}
            )
            self.assertEqual(result, mock_dynamic_tracer)

    def test_get_tracer_to_use_for_request_without_dynamic_headers(self):
        """Test that get_tracer_to_use_for_request returns the default tracer when no dynamic headers are present."""
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()

        # Mock the dynamic header extraction to return None
        with patch.object(
            otel, "_get_dynamic_otel_headers_from_kwargs"
        ) as mock_get_headers:
            mock_get_headers.return_value = None

            kwargs = {}
            result = otel.get_tracer_to_use_for_request(kwargs)

            # Assertions
            mock_get_headers.assert_called_once_with(kwargs)
            self.assertEqual(result, otel.tracer)

    def test_get_dynamic_otel_headers_from_kwargs(self):
        """Test that _get_dynamic_otel_headers_from_kwargs correctly extracts dynamic headers from kwargs."""
        # Setup
        otel = OpenTelemetry()

        # Mock the construct_dynamic_otel_headers method
        with patch.object(otel, "construct_dynamic_otel_headers") as mock_construct:
            # Test case 1: With standard_callback_dynamic_params
            mock_construct.return_value = {
                "arize-space-id": "test-space",
                "api_key": "test-key",
            }

            standard_params = {
                "arize_space_key": "test-space",
                "arize_api_key": "test-key",
            }
            kwargs = {"standard_callback_dynamic_params": standard_params}

            result = otel._get_dynamic_otel_headers_from_kwargs(kwargs)

            # Assertions
            mock_construct.assert_called_once_with(
                standard_callback_dynamic_params=standard_params
            )
            self.assertEqual(
                result, {"arize-space-id": "test-space", "api_key": "test-key"}
            )

            # Test case 2: Without standard_callback_dynamic_params
            kwargs_empty = {}
            result_empty = otel._get_dynamic_otel_headers_from_kwargs(kwargs_empty)

            # Should return None when no dynamic params
            self.assertIsNone(result_empty)

            # Test case 3: With empty construct result
            mock_construct.return_value = {}
            result_empty_construct = otel._get_dynamic_otel_headers_from_kwargs(kwargs)

            # Should return None when construct returns empty dict
            self.assertIsNone(result_empty_construct)

    @patch("opentelemetry.sdk.trace.TracerProvider")
    @patch("opentelemetry.sdk.resources.Resource")
    def test_get_tracer_with_dynamic_headers(self, mock_resource, mock_tracer_provider):
        """Test that _get_tracer_with_dynamic_headers creates a temporary tracer with dynamic headers."""
        # Setup
        otel = OpenTelemetry()

        # Mock the span processor creation
        with patch.object(otel, "_get_span_processor") as mock_get_span_processor:
            mock_span_processor = MagicMock()
            mock_get_span_processor.return_value = mock_span_processor

            # Mock the tracer provider and its methods
            mock_provider_instance = MagicMock()
            mock_tracer_provider.return_value = mock_provider_instance
            mock_tracer = MagicMock()
            mock_provider_instance.get_tracer.return_value = mock_tracer

            # Mock the resource
            mock_resource_instance = MagicMock()
            mock_resource.return_value = mock_resource_instance

            # Test
            dynamic_headers = {"arize-space-id": "test-space", "api_key": "test-key"}
            result = otel._get_tracer_with_dynamic_headers(dynamic_headers)

            # Assertions
            mock_get_span_processor.assert_called_once_with(
                dynamic_headers=dynamic_headers
            )
            mock_provider_instance.add_span_processor.assert_called_once_with(
                mock_span_processor
            )
            mock_provider_instance.get_tracer.assert_called_once_with("litellm")
            self.assertEqual(result, mock_tracer)

    @patch.dict(os.environ, {}, clear=True)
    @patch("opentelemetry.sdk.resources.Resource.create")
    @patch("opentelemetry.sdk.resources.OTELResourceDetector")
    def test_get_litellm_resource_with_defaults(
        self, mock_detector_cls, mock_resource_create
    ):
        """Test _get_litellm_resource with default values when no environment variables are set."""
        # Mock the Resource.create method
        mock_base_resource = MagicMock()
        mock_resource_create.return_value = mock_base_resource

        # Mock the OTELResourceDetector
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_env_resource = MagicMock()
        mock_detector.detect.return_value = mock_env_resource

        # Mock the merged resource
        mock_merged_resource = MagicMock()
        mock_base_resource.merge.return_value = mock_merged_resource

        config = OpenTelemetryConfig()
        result = OpenTelemetry._get_litellm_resource(config)

        # Verify Resource.create was called with correct default attributes
        expected_attributes = {
            "service.name": "litellm",
            "deployment.environment": "production",
            "model_id": "litellm",
        }
        mock_resource_create.assert_called_once_with(expected_attributes)
        mock_detector.detect.assert_called_once()
        mock_base_resource.merge.assert_called_once_with(mock_env_resource)
        self.assertEqual(result, mock_merged_resource)

    @patch.dict(
        os.environ,
        {
            "OTEL_SERVICE_NAME": "test-service",
            "OTEL_ENVIRONMENT_NAME": "staging",
            "OTEL_MODEL_ID": "test-model",
        },
        clear=True,
    )
    @patch("opentelemetry.sdk.resources.Resource.create")
    @patch("opentelemetry.sdk.resources.OTELResourceDetector")
    def test_get_litellm_resource_with_litellm_env_vars(
        self, mock_detector_cls, mock_resource_create
    ):
        """Test _get_litellm_resource with LiteLLM-specific environment variables."""
        # Mock the Resource.create method
        mock_base_resource = MagicMock()
        mock_resource_create.return_value = mock_base_resource

        # Mock the OTELResourceDetector
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_env_resource = MagicMock()
        mock_detector.detect.return_value = mock_env_resource

        # Mock the merged resource
        mock_merged_resource = MagicMock()
        mock_base_resource.merge.return_value = mock_merged_resource

        config = OpenTelemetryConfig.from_env()
        result = OpenTelemetry._get_litellm_resource(config)

        # Verify Resource.create was called with environment variable values
        expected_attributes = {
            "service.name": "test-service",
            "deployment.environment": "staging",
            "model_id": "test-model",
        }
        mock_resource_create.assert_called_once_with(expected_attributes)
        mock_detector.detect.assert_called_once()
        mock_base_resource.merge.assert_called_once_with(mock_env_resource)
        self.assertEqual(result, mock_merged_resource)

    @patch.dict(
        os.environ,
        {
            "OTEL_RESOURCE_ATTRIBUTES": "service.name=otel-service,deployment.environment=production,custom.attr=value",
            "OTEL_SERVICE_NAME": "should-be-overridden",
        },
        clear=True,
    )
    @patch("opentelemetry.sdk.resources.Resource.create")
    @patch("opentelemetry.sdk.resources.OTELResourceDetector")
    def test_get_litellm_resource_with_otel_resource_attributes(
        self, mock_detector_cls, mock_resource_create
    ):
        """Test _get_litellm_resource with OTEL_RESOURCE_ATTRIBUTES environment variable."""
        # Mock the Resource.create method to simulate the actual behavior
        # In reality, Resource.create() would parse OTEL_RESOURCE_ATTRIBUTES and merge it
        mock_base_resource = MagicMock()
        mock_resource_create.return_value = mock_base_resource

        # Mock the OTELResourceDetector
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_env_resource = MagicMock()
        mock_detector.detect.return_value = mock_env_resource

        # Mock the merged resource
        mock_merged_resource = MagicMock()
        mock_base_resource.merge.return_value = mock_merged_resource

        config = OpenTelemetryConfig.from_env()
        result = OpenTelemetry._get_litellm_resource(config)

        # Verify Resource.create was called with the base attributes
        # The actual OTEL_RESOURCE_ATTRIBUTES parsing is handled by OpenTelemetry SDK
        expected_attributes = {
            "service.name": "should-be-overridden",
            "deployment.environment": "production",
            "model_id": "should-be-overridden",
        }
        mock_resource_create.assert_called_once_with(expected_attributes)
        mock_detector.detect.assert_called_once()
        mock_base_resource.merge.assert_called_once_with(mock_env_resource)
        self.assertEqual(result, mock_merged_resource)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_litellm_resource_integration_with_real_resource(self):
        """Integration test to verify _get_litellm_resource works with actual OpenTelemetry Resource."""
        config = OpenTelemetryConfig()
        result = OpenTelemetry._get_litellm_resource(config)

        # Verify the result is a Resource instance
        from opentelemetry.sdk.resources import Resource

        self.assertIsInstance(result, Resource)

        # Verify the resource has the expected default attributes
        attributes = result.attributes
        self.assertEqual(attributes.get("service.name"), "litellm")
        self.assertEqual(attributes.get("deployment.environment"), "production")
        self.assertEqual(attributes.get("model_id"), "litellm")

    @patch.dict(
        os.environ,
        {
            "OTEL_RESOURCE_ATTRIBUTES": "service.name=from-env,custom.attribute=test-value,deployment.environment=test-env"
        },
        clear=True,
    )
    def test_get_litellm_resource_real_otel_resource_attributes(self):
        """Integration test to verify OTEL_RESOURCE_ATTRIBUTES is properly handled."""
        config = OpenTelemetryConfig.from_env()
        result = OpenTelemetry._get_litellm_resource(config)

        print("RESULT", result)

        # Verify the result is a Resource instance
        from opentelemetry.sdk.resources import Resource

        self.assertIsInstance(result, Resource)

        # Verify that OTEL_RESOURCE_ATTRIBUTES values override the defaults
        attributes = result.attributes
        self.assertEqual(attributes.get("service.name"), "from-env")
        self.assertEqual(attributes.get("deployment.environment"), "test-env")
        self.assertEqual(attributes.get("custom.attribute"), "test-value")
        # model_id should still be set from the base attributes since it wasn't in OTEL_RESOURCE_ATTRIBUTES
        self.assertEqual(attributes.get("model_id"), "litellm")

    @patch.dict(
        os.environ,
        {
            "OTEL_SERVICE_NAME": "litellm-service",
            "OTEL_RESOURCE_ATTRIBUTES": "service.name=otel-override,extra.attr=extra-value",
        },
        clear=True,
    )
    def test_get_litellm_resource_precedence(self):
        """Test that OTEL_SERVICE_NAME takes precedence over OTEL_RESOURCE_ATTRIBUTES according to OpenTelemetry spec."""
        config = OpenTelemetryConfig.from_env()
        result = OpenTelemetry._get_litellm_resource(config)

        # Verify the result is a Resource instance
        from opentelemetry.sdk.resources import Resource

        self.assertIsInstance(result, Resource)

        # According to OpenTelemetry spec, OTEL_SERVICE_NAME takes precedence over service.name in OTEL_RESOURCE_ATTRIBUTES
        attributes = result.attributes
        self.assertEqual(attributes.get("service.name"), "litellm-service")
        # But other attributes from OTEL_RESOURCE_ATTRIBUTES should still be present
        self.assertEqual(attributes.get("extra.attr"), "extra-value")

    def test_handle_success_spans_only(self):
        # make sure neither events nor metrics is on
        os.environ.pop("LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS", None)
        os.environ.pop("LITELLM_OTEL_INTEGRATION_ENABLE_METRICS", None)

        # ─── build in‐memory OTEL providers/exporters ─────────────────────────────
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

        # no logs / no metrics
        log_exporter = InMemoryLogExporter()
        logger_provider = OTLoggerProvider()
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        # ─── instantiate our OpenTelemetry logger with test providers ───────────
        otel = OpenTelemetry(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,  # pass even if events disabled (safe)
        )
        # bind our tracer to the test tracer provider (global registration is a no-op after the first time)
        otel.tracer = tracer_provider.get_tracer(__name__)

        # ─── minimal input / output for a chat call ──────────────────────────────
        start = datetime.utcnow()
        end = start + timedelta(seconds=1)
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_kwargs.json")
        ) as f:
            kwargs = json.load(f)
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_response.json")
        ) as f:
            response_obj = json.load(f)

        # ─── exercise the hook ───────────────────────────────────────────────────
        otel._handle_success(kwargs, response_obj, start, end)

        # ─── assert spans only ───────────────────────────────────────────────────
        spans = span_exporter.get_finished_spans()
        self.assertTrue(spans, "Expected at least one span")
        # must have the top‐level litellm_request span
        # self.assertIn(
        #     LITELLM_REQUEST_SPAN_NAME,
        #     [s.name for s in spans],
        #     "litellm_request span missing",
        # )
        # model attribute should be on that span
        found = any(
            s.attributes and s.attributes.get("gen_ai.request.model") == self.MODEL
            for s in spans
        )
        self.assertTrue(found, "expected gen_ai.request.model on span attributes")

        # no metrics recorded
        self.assertIsNone(
            self.wait_for_metric(metric_reader, "gen_ai.client.operation.duration"),
            "Did not expect any metrics",
        )
        # no logs emitted
        logs = log_exporter.get_finished_logs()
        self.assertFalse(logs, "Did not expect any logs")

    @patch.dict(
        os.environ, {"LITELLM_OTEL_INTEGRATION_ENABLE_METRICS": "true"}, clear=True
    )
    def test_handle_success_spans_and_metrics(self):
        # ─── build in‐memory OTEL providers/exporters ─────────────────────────────
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

        log_exporter = InMemoryLogExporter()
        logger_provider = OTLoggerProvider()
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        # ─── instantiate our OpenTelemetry logger with test providers ───────────
        otel = OpenTelemetry(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,  # needed if events were enabled
        )
        otel.tracer = tracer_provider.get_tracer(__name__)

        # ─── minimal input / output for a chat call ──────────────────────────────
        start = datetime.utcnow()
        end = start + timedelta(seconds=1)
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_kwargs.json")
        ) as f:
            kwargs = json.load(f)
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_response.json")
        ) as f:
            response_obj = json.load(f)

        # ─── exercise the hook ───────────────────────────────────────────────────
        otel._handle_success(kwargs, response_obj, start, end)

        # ─── assert spans ────────────────────────────────────────────────────────
        spans = span_exporter.get_finished_spans()
        self.assertTrue(spans, "Expected at least one span")

        # ─── assert metrics ──────────────────────────────────────────────────────
        duration_metric = self.wait_for_metric(
            metric_reader, "gen_ai.client.operation.duration"
        )
        self.assertIsNotNone(duration_metric, "duration histogram was not recorded")
        # model attribute should be present on a data point
        found_dp = False
        if (
            duration_metric
            and hasattr(duration_metric, "data")
            and hasattr(duration_metric.data, "data_points")
        ):
            found_dp = any(
                dp.attributes.get("gen_ai.request.model") == self.MODEL
                for dp in duration_metric.data.data_points
            )
        self.assertTrue(
            found_dp, "expected gen_ai.request.model attribute on a data point"
        )

        # ─── no events when only metrics enabled ─────────────────────────────────
        logs = log_exporter.get_finished_logs()
        self.assertFalse(logs, "Did not expect any logs")

    def test_get_span_name_with_generation_name(self):
        """Test _get_span_name returns generation_name when present"""
        otel = OpenTelemetry()
        kwargs = {"litellm_params": {"metadata": {"generation_name": "custom_span"}}}
        result = otel._get_span_name(kwargs)
        self.assertEqual(result, "custom_span")

    def test_get_span_name_without_generation_name(self):
        """Test _get_span_name returns default when generation_name missing"""
        from litellm.integrations.opentelemetry import LITELLM_REQUEST_SPAN_NAME

        otel = OpenTelemetry()
        kwargs = {"litellm_params": {"metadata": {}}}
        result = otel._get_span_name(kwargs)
        self.assertEqual(result, LITELLM_REQUEST_SPAN_NAME)

    @patch("litellm.turn_off_message_logging", False)
    def test_maybe_log_raw_request_creates_span(self):
        """Test _maybe_log_raw_request creates span when logging enabled"""
        from litellm.integrations.opentelemetry import RAW_REQUEST_SPAN_NAME

        otel = OpenTelemetry()
        otel.message_logging = True

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        otel.get_tracer_to_use_for_request = MagicMock(return_value=mock_tracer)
        otel.set_raw_request_attributes = MagicMock()
        otel._to_ns = MagicMock(return_value=1234567890)

        kwargs = {"litellm_params": {"metadata": {}}}
        otel._maybe_log_raw_request(
            kwargs, {}, datetime.now(), datetime.now(), MagicMock()
        )

        mock_tracer.start_span.assert_called_once()
        self.assertEqual(
            mock_tracer.start_span.call_args[1]["name"], RAW_REQUEST_SPAN_NAME
        )

    @patch("litellm.turn_off_message_logging", True)
    def test_maybe_log_raw_request_skips_when_logging_disabled(self):
        """Test _maybe_log_raw_request skips when logging disabled"""
        otel = OpenTelemetry()
        mock_tracer = MagicMock()
        otel.get_tracer_to_use_for_request = MagicMock(return_value=mock_tracer)

        kwargs = {"litellm_params": {"metadata": {}}}
        otel._maybe_log_raw_request(
            kwargs, {}, datetime.now(), datetime.now(), MagicMock()
        )

        mock_tracer.start_span.assert_not_called()


class TestOpenTelemetryHeaderSplitting(unittest.TestCase):
    """Test suite for _get_headers_dictionary method"""

    def test_split_multiple_headers_comma_separated(self):
        """Test splitting multiple headers separated by commas"""
        otel = OpenTelemetry()
        headers = "api-key=key,other-config-value=value"
        result = otel._get_headers_dictionary(headers)
        self.assertEqual(result, {"api-key": "key", "other-config-value": "value"})

    def test_split_headers_with_equals_in_values(self):
        """Test splitting headers where values contain equals signs (split only on first '=')"""
        otel = OpenTelemetry()
        headers = "api-key=value1=part2,config=setting=enabled"
        result = otel._get_headers_dictionary(headers)
        self.assertEqual(
            result, {"api-key": "value1=part2", "config": "setting=enabled"}
        )


class TestOpenTelemetryEndpointNormalization(unittest.TestCase):
    """Test suite for the unified _normalize_otel_endpoint method"""

    def test_normalize_traces_endpoint_from_logs_path(self):
        """Test normalizing endpoint with /v1/logs to /v1/traces"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/logs", "traces"
        )
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_traces_endpoint_from_metrics_path(self):
        """Test normalizing endpoint with /v1/metrics to /v1/traces"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/metrics", "traces"
        )
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_traces_endpoint_from_base_url(self):
        """Test adding /v1/traces to base URL"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318", "traces")
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_traces_endpoint_from_v1_path(self):
        """Test adding traces to /v1 path"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318/v1", "traces")
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_traces_endpoint_already_correct(self):
        """Test endpoint already ending with /v1/traces remains unchanged"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/traces", "traces"
        )
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_metrics_endpoint_from_traces_path(self):
        """Test normalizing endpoint with /v1/traces to /v1/metrics"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/traces", "metrics"
        )
        self.assertEqual(result, "http://collector:4318/v1/metrics")

    def test_normalize_metrics_endpoint_from_logs_path(self):
        """Test normalizing endpoint with /v1/logs to /v1/metrics"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/logs", "metrics"
        )
        self.assertEqual(result, "http://collector:4318/v1/metrics")

    def test_normalize_metrics_endpoint_from_base_url(self):
        """Test adding /v1/metrics to base URL"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318", "metrics")
        self.assertEqual(result, "http://collector:4318/v1/metrics")

    def test_normalize_metrics_endpoint_already_correct(self):
        """Test endpoint already ending with /v1/metrics remains unchanged"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/metrics", "metrics"
        )
        self.assertEqual(result, "http://collector:4318/v1/metrics")

    def test_normalize_logs_endpoint_from_traces_path(self):
        """Test normalizing endpoint with /v1/traces to /v1/logs"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/traces", "logs"
        )
        self.assertEqual(result, "http://collector:4318/v1/logs")

    def test_normalize_logs_endpoint_from_metrics_path(self):
        """Test normalizing endpoint with /v1/metrics to /v1/logs"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/metrics", "logs"
        )
        self.assertEqual(result, "http://collector:4318/v1/logs")

    def test_normalize_logs_endpoint_from_base_url(self):
        """Test adding /v1/logs to base URL"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318", "logs")
        self.assertEqual(result, "http://collector:4318/v1/logs")

    def test_normalize_logs_endpoint_already_correct(self):
        """Test endpoint already ending with /v1/logs remains unchanged"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318/v1/logs", "logs")
        self.assertEqual(result, "http://collector:4318/v1/logs")

    def test_normalize_endpoint_with_trailing_slash(self):
        """Test that trailing slashes are properly handled"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318/", "traces")
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_endpoint_none(self):
        """Test that None endpoint returns None"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(None, "traces")
        self.assertIsNone(result)

    def test_normalize_endpoint_empty_string(self):
        """Test that empty string returns empty string"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("", "traces")
        self.assertEqual(result, "")

    def test_normalize_endpoint_invalid_signal_type(self):
        """Test that invalid signal type returns endpoint unchanged with warning"""
        otel = OpenTelemetry()
        endpoint = "http://collector:4318/v1/traces"

        with patch("litellm._logging.verbose_logger.warning") as mock_warning:
            result = otel._normalize_otel_endpoint(endpoint, "invalid")

            # Should return endpoint unchanged
            self.assertEqual(result, endpoint)

            # Should log a warning
            mock_warning.assert_called_once()
            # Check the warning was called with the expected format string and parameters
            call_args = mock_warning.call_args[0]
            self.assertIn("Invalid signal_type", call_args[0])
            self.assertEqual(call_args[1], "invalid")  # signal_type parameter
            self.assertEqual(
                call_args[2], {"traces", "metrics", "logs"}
            )  # valid_signals parameter

    def test_normalize_endpoint_https(self):
        """Test normalization works with https URLs"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "https://collector.example.com:4318", "logs"
        )
        self.assertEqual(result, "https://collector.example.com:4318/v1/logs")

    def test_normalize_endpoint_with_path_prefix(self):
        """Test normalization works with URLs that have path prefixes"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/otel/v1/traces", "logs"
        )
        # Should replace the final /traces with /logs
        self.assertEqual(result, "http://collector:4318/otel/v1/logs")

    def test_normalize_endpoint_consistency_across_signals(self):
        """Test that normalization is consistent for all signal types from the same base"""
        otel = OpenTelemetry()
        base = "http://collector:4318"

        traces_result = otel._normalize_otel_endpoint(base, "traces")
        metrics_result = otel._normalize_otel_endpoint(base, "metrics")
        logs_result = otel._normalize_otel_endpoint(base, "logs")

        # All should have the same base with different signal paths
        self.assertEqual(traces_result, "http://collector:4318/v1/traces")
        self.assertEqual(metrics_result, "http://collector:4318/v1/metrics")
        self.assertEqual(logs_result, "http://collector:4318/v1/logs")

    def test_normalize_endpoint_signal_switching(self):
        """Test switching between different signal types on the same endpoint"""
        otel = OpenTelemetry()

        # Start with traces
        endpoint = "http://collector:4318/v1/traces"

        # Switch to metrics
        metrics = otel._normalize_otel_endpoint(endpoint, "metrics")
        self.assertEqual(metrics, "http://collector:4318/v1/metrics")

        # Switch to logs
        logs = otel._normalize_otel_endpoint(metrics, "logs")
        self.assertEqual(logs, "http://collector:4318/v1/logs")

        # Switch back to traces
        traces = otel._normalize_otel_endpoint(logs, "traces")
        self.assertEqual(traces, "http://collector:4318/v1/traces")


class TestOpenTelemetryProtocolSelection(unittest.TestCase):
    """Test suite for verifying correct exporter selection based on protocol"""

    def test_get_span_processor_uses_http_exporter_for_otlp_http(self):
        """Test that otlp_http protocol uses OTLPSpanExporterHTTP"""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig(
            exporter="otlp_http", endpoint="http://collector:4318"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the HTTP variant
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterHTTP)

    def test_get_span_processor_uses_grpc_exporter_for_otlp_grpc(self):
        """Test that otlp_grpc protocol uses OTLPSpanExporterGRPC"""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig(
            exporter="otlp_grpc", endpoint="http://collector:4317"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the gRPC variant
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterGRPC)

    def test_get_span_processor_uses_grpc_exporter_for_grpc_alias(self):
        """Test that 'grpc' protocol alias uses OTLPSpanExporterGRPC"""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig(exporter="grpc", endpoint="http://collector:4317")
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the gRPC variant
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterGRPC)

    def test_get_span_processor_uses_http_exporter_for_http_protobuf(self):
        """Test that http/protobuf protocol uses OTLPSpanExporterHTTP"""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig(
            exporter="http/protobuf", endpoint="http://collector:4318"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the HTTP variant
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterHTTP)

    def test_get_span_processor_uses_console_exporter_for_console(self):
        """Test that console protocol uses ConsoleSpanExporter"""
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        config = OpenTelemetryConfig(exporter="console")
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the console variant
        self.assertIsInstance(processor.span_exporter, ConsoleSpanExporter)

    def test_get_log_exporter_uses_http_exporter_for_otlp_http(self):
        """Test that otlp_http protocol uses HTTP OTLPLogExporter"""
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        config = OpenTelemetryConfig(
            exporter="otlp_http", endpoint="http://collector:4318", enable_events=True
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the exporter is the HTTP variant
        self.assertIsInstance(exporter, OTLPLogExporter)

        # Check that it's from the http module by checking the module name
        self.assertIn("http", exporter.__class__.__module__)

    def test_get_log_exporter_uses_grpc_exporter_for_otlp_grpc(self):
        """Test that otlp_grpc protocol uses gRPC OTLPLogExporter"""
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        config = OpenTelemetryConfig(
            exporter="otlp_grpc", endpoint="http://collector:4317", enable_events=True
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the exporter is the gRPC variant
        self.assertIsInstance(exporter, OTLPLogExporter)

        # Check that it's from the grpc module by checking the module name
        self.assertIn("grpc", exporter.__class__.__module__)

    def test_get_log_exporter_uses_grpc_exporter_for_grpc_alias(self):
        """Test that 'grpc' protocol alias uses gRPC OTLPLogExporter"""
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        config = OpenTelemetryConfig(
            exporter="grpc", endpoint="http://collector:4317", enable_events=True
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the exporter is the gRPC variant
        self.assertIsInstance(exporter, OTLPLogExporter)

        # Check that it's from the grpc module by checking the module name
        self.assertIn("grpc", exporter.__class__.__module__)

    def test_get_log_exporter_uses_console_exporter_for_console(self):
        """Test that console protocol uses ConsoleLogExporter"""
        from opentelemetry.sdk._logs.export import ConsoleLogExporter

        config = OpenTelemetryConfig(exporter="console", enable_events=True)
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the exporter is the console variant
        self.assertIsInstance(exporter, ConsoleLogExporter)

    def test_get_log_exporter_defaults_to_console_for_unknown_protocol(self):
        """Test that unknown protocol defaults to ConsoleLogExporter with warning"""
        from opentelemetry.sdk._logs.export import ConsoleLogExporter

        config = OpenTelemetryConfig(exporter="unknown_protocol", enable_events=True)
        otel = OpenTelemetry(config=config)

        with patch("litellm._logging.verbose_logger.warning") as mock_warning:
            exporter = otel._get_log_exporter()

            # Verify the exporter defaults to console
            self.assertIsInstance(exporter, ConsoleLogExporter)

            # Verify a warning was logged
            mock_warning.assert_called_once()
            args = mock_warning.call_args[0]
            self.assertIn("Unknown log exporter", args[0])
            self.assertIn("unknown_protocol", args[1])

    @patch.dict(
        os.environ,
        {
            "OTEL_EXPORTER": "otlp_http",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318",
        },
        clear=False,
    )
    def test_protocol_selection_from_environment_http(self):
        """Test that protocol selection works correctly from environment variables for HTTP"""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig.from_env()
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify the HTTP exporter is used
        self.assertIsInstance(processor, BatchSpanProcessor)
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterHTTP)

    @patch.dict(
        os.environ,
        {
            "OTEL_EXPORTER": "otlp_grpc",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317",
        },
        clear=False,
    )
    def test_protocol_selection_from_environment_grpc(self):
        """Test that protocol selection works correctly from environment variables for gRPC"""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig.from_env()
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify the gRPC exporter is used
        self.assertIsInstance(processor, BatchSpanProcessor)
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterGRPC)

    def test_http_exporter_endpoint_normalization_for_traces(self):
        """Test that HTTP trace exporter gets properly normalized endpoint"""
        config = OpenTelemetryConfig(
            exporter="otlp_http", endpoint="http://collector:4318"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify the endpoint was normalized to include /v1/traces
        # Access the private _endpoint attribute if available
        if hasattr(processor.span_exporter, "_endpoint"):
            self.assertEqual(processor.span_exporter._endpoint, "http://collector:4318/v1/traces")  # type: ignore[attr-defined]

    def test_grpc_exporter_endpoint_normalization_for_traces(self):
        """Test that gRPC trace exporter gets properly normalized endpoint"""
        config = OpenTelemetryConfig(
            exporter="otlp_grpc", endpoint="http://collector:4317"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify the endpoint was normalized to include /v1/traces
        # Note: gRPC exporters strip the http:// prefix, so we check for the normalized path
        if hasattr(processor.span_exporter, "_endpoint"):
            # gRPC exporter strips http:// prefix
            self.assertIn("collector:4317", processor.span_exporter._endpoint)  # type: ignore[attr-defined]
            # The endpoint should have been normalized with /v1/traces before being passed to gRPC exporter
            # We verify this by checking the normalization function was called correctly
            normalized = otel._normalize_otel_endpoint(
                "http://collector:4317", "traces"
            )
            self.assertEqual(normalized, "http://collector:4317/v1/traces")

    def test_http_log_exporter_endpoint_normalization_for_logs(self):
        """Test that HTTP log exporter gets properly normalized endpoint"""
        config = OpenTelemetryConfig(
            exporter="otlp_http",
            endpoint="http://collector:4318/v1/traces",
            enable_events=True,
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the endpoint was normalized to /v1/logs (not /v1/traces)
        # Access the private _endpoint attribute if available
        if hasattr(exporter, "_endpoint"):
            self.assertEqual(exporter._endpoint, "http://collector:4318/v1/logs")  # type: ignore[attr-defined]

    def test_grpc_log_exporter_endpoint_normalization_for_logs(self):
        """Test that gRPC log exporter gets properly normalized endpoint"""
        config = OpenTelemetryConfig(
            exporter="otlp_grpc",
            endpoint="http://collector:4317/v1/traces",
            enable_events=True,
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the endpoint was normalized to /v1/logs (not /v1/traces)
        # Note: gRPC exporters strip the http:// prefix, so we check for the normalized path
        if hasattr(exporter, "_endpoint"):
            # gRPC exporter strips http:// prefix
            self.assertIn("collector:4317", exporter._endpoint)  # type: ignore[attr-defined]
            # The endpoint should have been normalized with /v1/logs before being passed to gRPC exporter
            # We verify this by checking the normalization function was called correctly
            normalized = otel._normalize_otel_endpoint(
                "http://collector:4317/v1/traces", "logs"
            )
            self.assertEqual(normalized, "http://collector:4317/v1/logs")

    def test_get_metric_reader_uses_http_exporter_for_http_protobuf(self):
        """Test that http/protobuf protocol uses OTLPMetricExporterHTTP"""
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        config = OpenTelemetryConfig(
            exporter="http/protobuf", endpoint="http://collector:4318"
        )
        otel = OpenTelemetry(config=config)

        reader = otel._get_metric_reader()

        self.assertIsInstance(reader, PeriodicExportingMetricReader)
        self.assertIsInstance(reader._exporter, OTLPMetricExporter)


class TestOpenTelemetryExternalSpan(unittest.TestCase):
    """
    Test suite for external span handling in OpenTelemetry integration.

    These tests verify that LiteLLM correctly handles spans created outside
    of LiteLLM (e.g., by Langfuse SDK, user application code, or global context)
    without closing them prematurely.

    Background:
    - External spans can come from: Langfuse SDK, user code, HTTP traceparent headers, global context
    - LiteLLM should NEVER close spans it did not create
    - Bug: LiteLLM was reusing and closing external spans in _start_primary_span
    """

    HERE = os.path.dirname(__file__)

    def setUp(self):
        """Set up common test fixtures"""
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))

        # Don't set global tracer provider - instead, get tracers directly from our provider
        # This avoids "Overriding of current TracerProvider is not allowed" warnings

        # Clear any existing spans
        self.span_exporter.clear()

    def _create_test_kwargs_and_response(self):
        """Load test data from JSON files"""
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_kwargs.json")
        ) as f:
            kwargs = json.load(f)

        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_response.json")
        ) as f:
            response_obj = json.load(f)

        return kwargs, response_obj

    def _get_spans_by_name(self, name):
        """Get all spans with the given name"""
        spans = self.span_exporter.get_finished_spans()
        return [s for s in spans if s.name == name]

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_not_closed_with_use_otel_litellm_request_span_false(self):
        """
        Test that external spans are not closed when USE_OTEL_LITELLM_REQUEST_SPAN=false (default).

        Expected behavior:
        - External span remains open (is_recording = True)
        - raw_gen_ai_request spans are direct children of external span (shallow hierarchy)
        - No litellm_request span is created
        - Multiple completions work correctly
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id
            parent_span_id = parent_ctx.span_id

            self.assertTrue(
                parent_span.is_recording(),
                "External span should be recording before completion calls",
            )

            # First completion call
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time, end_time)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording after first completion",
            )

            # Second completion call
            start_time2 = end_time
            end_time2 = start_time2 + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time2, end_time2)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording after second completion",
            )

        # After exiting context, verify spans
        spans = self.span_exporter.get_finished_spans()

        # All spans should have the same trace_id
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                f"Span {span.name} should have same trace_id as parent",
            )

        # Should have external_parent_span
        parent_spans = self._get_spans_by_name("external_parent_span")
        self.assertEqual(
            len(parent_spans), 1, "Should have exactly one external_parent_span"
        )

        # Verify LiteLLM set attributes on external parent span
        parent_span_finished = parent_spans[0]
        self.assertIsNotNone(
            parent_span_finished.attributes,
            "Parent span should have attributes set by LiteLLM",
        )
        self.assertIn(
            "gen_ai.request.model",
            parent_span_finished.attributes,
            "Parent span should have model attribute from LiteLLM",
        )

        # Should have raw_gen_ai_request spans (if message_logging is on)
        raw_spans = self._get_spans_by_name("raw_gen_ai_request")
        # Note: May be 0 if message_logging is off, or 2 if on

        # Should NOT have litellm_request spans (USE_OTEL_LITELLM_REQUEST_SPAN=false)
        litellm_spans = self._get_spans_by_name("litellm_request")
        self.assertEqual(
            len(litellm_spans),
            0,
            "Should NOT have litellm_request spans when USE_OTEL_LITELLM_REQUEST_SPAN=false",
        )

        # Verify raw_gen_ai_request spans are direct children of external span
        for raw_span in raw_spans:
            self.assertEqual(
                raw_span.parent.span_id if raw_span.parent else None,
                parent_span_id,
                f"raw_gen_ai_request should be direct child of external_parent_span",
            )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "true"}, clear=False)
    def test_external_span_not_closed_with_use_otel_litellm_request_span_true(self):
        """
        Test that external spans are not closed when USE_OTEL_LITELLM_REQUEST_SPAN=true.

        Expected behavior:
        - External span remains open (is_recording = True)
        - litellm_request spans are created as children of external span
        - raw_gen_ai_request spans are children of litellm_request spans
        - Correct hierarchy: external_parent → litellm_request → raw_gen_ai_request
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id
            parent_span_id = parent_ctx.span_id

            # First completion call
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time, end_time)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording after first completion",
            )

            # Second completion call
            start_time2 = end_time
            end_time2 = start_time2 + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time2, end_time2)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording after second completion",
            )

        # After exiting context, verify spans
        spans = self.span_exporter.get_finished_spans()

        # All spans should have the same trace_id
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                f"Span {span.name} should have same trace_id as parent",
            )

        # Should have litellm_request spans (USE_OTEL_LITELLM_REQUEST_SPAN=true)
        litellm_spans = self._get_spans_by_name("litellm_request")
        self.assertEqual(
            len(litellm_spans),
            2,
            "Should have 2 litellm_request spans when USE_OTEL_LITELLM_REQUEST_SPAN=true",
        )

        # Verify litellm_request spans are children of external span
        for litellm_span in litellm_spans:
            self.assertEqual(
                litellm_span.parent.span_id if litellm_span.parent else None,
                parent_span_id,
                "litellm_request should be child of external_parent_span",
            )

        # Verify raw_gen_ai_request spans (if present) are children of litellm_request
        raw_spans = self._get_spans_by_name("raw_gen_ai_request")
        if raw_spans:
            litellm_span_ids = {s.context.span_id for s in litellm_spans}
            for raw_span in raw_spans:
                self.assertIn(
                    raw_span.parent.span_id if raw_span.parent else None,
                    litellm_span_ids,
                    "raw_gen_ai_request should be child of litellm_request",
                )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_with_multiple_completions(self):
        """
        Test that multiple completion calls work correctly within external span context.

        Expected behavior:
        - Both completion calls succeed
        - All spans belong to the same trace
        - External span remains open throughout
        - No errors or warnings about "ended span"
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id

            # Make multiple completion calls
            for i in range(3):
                start_time = datetime.utcnow()
                end_time = start_time + timedelta(seconds=1)

                # This should not raise any exceptions
                otel._handle_success(kwargs, response_obj, start_time, end_time)

                # Verify parent span is still recording after each call
                self.assertTrue(
                    parent_span.is_recording(),
                    f"External span should still be recording after completion #{i+1}",
                )

        # Verify all spans have the same trace_id
        spans = self.span_exporter.get_finished_spans()
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                f"All spans should belong to the same trace",
            )

        # Should have the external parent span
        parent_spans = self._get_spans_by_name("external_parent_span")
        self.assertEqual(
            len(parent_spans), 1, "Should have exactly one external_parent_span"
        )

        # Verify LiteLLM set attributes on external parent span
        parent_span_finished = parent_spans[0]
        self.assertIn(
            "gen_ai.request.model",
            parent_span_finished.attributes,
            "Parent span should have model attribute from LiteLLM",
        )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_from_global_context(self):
        """
        Test external span detection from global context (Priority 3 in _get_span_context).

        This simulates the case where a span is set in the global context
        (e.g., by user code or Langfuse SDK) and LiteLLM detects it via
        trace.get_current_span().

        Expected behavior:
        - LiteLLM detects the span from global context
        - External span is not closed
        - Correct parent-child relationship
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span and set it as current using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_global_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id

            # Verify the span is in global context
            current_span = trace.get_current_span()
            self.assertEqual(
                current_span, parent_span, "Span should be in global context"
            )

            # Make completion call
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time, end_time)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span from global context should not be closed",
            )

        # Verify trace structure
        spans = self.span_exporter.get_finished_spans()
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                "All spans should have the same trace_id",
            )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_hierarchy_preserved(self):
        """
        Test that span hierarchy is correctly preserved with external parent.

        Expected behavior:
        - Parent span IDs are correct
        - Trace structure matches expected hierarchy
        - Span names are correct
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)
        otel.message_logging = (
            True  # Enable message logging to get raw_gen_ai_request spans
        )

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_span_id = parent_span.get_span_context().span_id

            # Make completion call
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time, end_time)

        # Verify hierarchy
        spans = self.span_exporter.get_finished_spans()

        # Get spans by name
        parent_spans = self._get_spans_by_name("external_parent_span")
        raw_spans = self._get_spans_by_name("raw_gen_ai_request")

        self.assertEqual(len(parent_spans), 1, "Should have one parent span")

        # Verify parent-child relationship
        if raw_spans:  # If message_logging is on
            for raw_span in raw_spans:
                self.assertEqual(
                    raw_span.parent.span_id if raw_span.parent else None,
                    parent_span_id,
                    "raw_gen_ai_request should be child of external_parent_span",
                )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_not_ended_on_failure(self):
        """
        Test that external spans are not closed even on failure.

        Expected behavior:
        - When _handle_failure is called with external span context
        - External span remains open (is_recording = True)
        - Error span is created correctly
        - External span status is NOT changed by LiteLLM
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id

            # Simulate failure
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)

            # Create error response object
            error_response = {"error": "Test error"}

            # Call _handle_failure
            otel._handle_failure(kwargs, error_response, start_time, end_time)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording even after failure",
            )

        # Verify trace structure
        spans = self.span_exporter.get_finished_spans()

        # All spans should have the same trace_id
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                "All spans should have the same trace_id even on failure",
            )

        # Should have external_parent_span
        parent_spans = self._get_spans_by_name("external_parent_span")
        self.assertEqual(
            len(parent_spans), 1, "Should have exactly one external_parent_span"
        )

        # Verify LiteLLM set attributes on external parent span even on failure
        parent_span_finished = parent_spans[0]
        self.assertIn(
            "gen_ai.request.model",
            parent_span_finished.attributes,
            "Parent span should have model attribute from LiteLLM even on failure",
        )


class TestOpenTelemetrySemanticConventions138(unittest.TestCase):
    """
    Test suite for OpenTelemetry 1.38 Semantic Conventions compliance.

    These tests verify that LiteLLM emits span attributes following the
    OpenTelemetry GenAI semantic conventions v1.38, including:
    - gen_ai.input.messages (JSON string with parts array)
    - gen_ai.output.messages (JSON string with parts array)
    - gen_ai.usage.input_tokens / output_tokens (new naming)
    - gen_ai.response.finish_reasons (JSON array)

    See: https://github.com/BerriAI/litellm/issues/17794
    """

    def test_input_messages_uses_parts_structure(self):
        """
        Test that gen_ai.input.messages uses the OTEL 1.38 parts array structure.

        Expected format:
        [{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello world"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hi there!"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Find the call that set gen_ai.input.messages
        input_messages_calls = [
            call
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "gen_ai.input.messages"
        ]
        self.assertEqual(
            len(input_messages_calls),
            1,
            "Should have exactly one gen_ai.input.messages attribute",
        )

        input_messages_value = input_messages_calls[0][0][1]
        parsed = json.loads(input_messages_value)

        # Verify structure
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["role"], "user")
        self.assertIn("parts", parsed[0])
        self.assertEqual(parsed[0]["parts"][0]["type"], "text")
        self.assertEqual(parsed[0]["parts"][0]["content"], "Hello world")

    def test_output_messages_uses_parts_structure(self):
        """
        Test that gen_ai.output.messages uses the OTEL 1.38 parts array structure.

        Expected format:
        [{"role": "assistant", "parts": [{"type": "text", "content": "Hi!"}], "finish_reason": "stop"}]
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello back!"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Find the call that set gen_ai.output.messages
        output_messages_calls = [
            call
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "gen_ai.output.messages"
        ]
        self.assertEqual(
            len(output_messages_calls),
            1,
            "Should have exactly one gen_ai.output.messages attribute",
        )

        output_messages_value = output_messages_calls[0][0][1]
        parsed = json.loads(output_messages_value)

        # Verify structure
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["role"], "assistant")
        self.assertIn("parts", parsed[0])
        self.assertEqual(parsed[0]["parts"][0]["type"], "text")
        self.assertEqual(parsed[0]["parts"][0]["content"], "Hello back!")
        self.assertEqual(parsed[0]["finish_reason"], "stop")

    def test_usage_tokens_use_new_naming_convention(self):
        """
        Test that token usage uses the OTEL 1.38 naming convention:
        - gen_ai.usage.input_tokens (not prompt_tokens)
        - gen_ai.usage.output_tokens (not completion_tokens)
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Verify new naming convention is used
        mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 100)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 50)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)

    def test_finish_reasons_is_json_array(self):
        """
        Test that gen_ai.response.finish_reasons is a proper JSON array.

        Expected: '["stop"]' (not "['stop']")
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hi"},
                },
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Find the call that set gen_ai.response.finish_reasons
        finish_reasons_calls = [
            call
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "gen_ai.response.finish_reasons"
        ]
        self.assertEqual(
            len(finish_reasons_calls),
            1,
            "Should have exactly one gen_ai.response.finish_reasons attribute",
        )

        finish_reasons_value = finish_reasons_calls[0][0][1]

        # Verify it's valid JSON (not Python repr)
        parsed = json.loads(finish_reasons_value)
        self.assertEqual(parsed, ["stop"])

    def test_operation_name_is_chat_for_completion(self):
        """
        Test that gen_ai.operation.name is 'chat' for completion calls.
        Regression guard: completion -> 'chat' must not regress.
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        call_args = {k: v for call in mock_span.set_attribute.call_args_list for k, v in [call[0]]}
        assert call_args.get("gen_ai.operation.name") == "chat", (
            f"Expected 'chat' for completion call_type, got {call_args.get('gen_ai.operation.name')!r}"
        )

    def test_operation_name_is_embeddings_for_embedding(self):
        """
        gen_ai.operation.name must be 'embeddings' for embedding call_type (not 'chat').
        Regression test for the hardcoded 'chat' bug fixed in commit 7fcd20c.
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "text-embedding-ada-002",
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "embedding",
                "metadata": {},
            },
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj={})

        call_args = {k: v for call in mock_span.set_attribute.call_args_list for k, v in [call[0]]}
        assert call_args.get("gen_ai.operation.name") == "embeddings", (
            f"Expected 'embeddings' for embedding call_type, got {call_args.get('gen_ai.operation.name')!r}"
        )

    def test_operation_name_is_embeddings_for_aembedding(self):
        """async embedding variant must also map to 'embeddings'."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "text-embedding-ada-002",
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "aembedding",
                "metadata": {},
            },
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj={})

        call_args = {k: v for call in mock_span.set_attribute.call_args_list for k, v in [call[0]]}
        assert call_args.get("gen_ai.operation.name") == "embeddings"

    def test_operation_name_for_transcription(self):
        """gen_ai.operation.name must be 'transcription' for transcription call_type."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "whisper-1",
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "transcription",
                "metadata": {},
            },
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj={})

        call_args = {k: v for call in mock_span.set_attribute.call_args_list for k, v in [call[0]]}
        assert call_args.get("gen_ai.operation.name") == "transcription"

    def test_operation_name_for_speech(self):
        """gen_ai.operation.name must be 'speech' for speech call_type."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "tts-1",
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "speech",
                "metadata": {},
            },
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj={})

        call_args = {k: v for call in mock_span.set_attribute.call_args_list for k, v in [call[0]]}
        assert call_args.get("gen_ai.operation.name") == "speech"

    def test_operation_name_for_rerank(self):
        """gen_ai.operation.name must be 'rerank' for rerank call_type."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "rerank-english-v2.0",
            "litellm_params": {"custom_llm_provider": "cohere"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "rerank",
                "metadata": {},
            },
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj={})

        call_args = {k: v for call in mock_span.set_attribute.call_args_list for k, v in [call[0]]}
        assert call_args.get("gen_ai.operation.name") == "rerank"


class TestMapCallTypeToOperationName(unittest.TestCase):
    """Unit tests for OpenTelemetry._map_call_type_to_operation_name (pure function)."""

    def test_completion_maps_to_chat(self):
        assert OpenTelemetry._map_call_type_to_operation_name("completion") == "chat"

    def test_acompletion_maps_to_chat(self):
        assert OpenTelemetry._map_call_type_to_operation_name("acompletion") == "chat"

    def test_text_completion_maps_to_text_completion(self):
        assert OpenTelemetry._map_call_type_to_operation_name("text_completion") == "text_completion"

    def test_atext_completion_maps_to_text_completion(self):
        assert OpenTelemetry._map_call_type_to_operation_name("atext_completion") == "text_completion"

    def test_embedding_maps_to_embeddings(self):
        assert OpenTelemetry._map_call_type_to_operation_name("embedding") == "embeddings"

    def test_aembedding_maps_to_embeddings(self):
        assert OpenTelemetry._map_call_type_to_operation_name("aembedding") == "embeddings"

    def test_image_generation(self):
        assert OpenTelemetry._map_call_type_to_operation_name("image_generation") == "image_generation"

    def test_aimage_generation(self):
        assert OpenTelemetry._map_call_type_to_operation_name("aimage_generation") == "image_generation"

    def test_transcription(self):
        assert OpenTelemetry._map_call_type_to_operation_name("transcription") == "transcription"

    def test_atranscription(self):
        assert OpenTelemetry._map_call_type_to_operation_name("atranscription") == "transcription"

    def test_speech(self):
        assert OpenTelemetry._map_call_type_to_operation_name("speech") == "speech"

    def test_aspeech(self):
        assert OpenTelemetry._map_call_type_to_operation_name("aspeech") == "speech"

    def test_rerank(self):
        assert OpenTelemetry._map_call_type_to_operation_name("rerank") == "rerank"

    def test_arerank(self):
        assert OpenTelemetry._map_call_type_to_operation_name("arerank") == "rerank"

    def test_unknown_type_passes_through(self):
        """Unknown call types should be returned unchanged as a safe fallback."""
        assert OpenTelemetry._map_call_type_to_operation_name("some_future_type") == "some_future_type"

    def test_all_async_variants_match_sync(self):
        """Every async variant must map to the same value as its sync counterpart."""
        pairs = [
            ("completion", "acompletion"),
            ("text_completion", "atext_completion"),
            ("embedding", "aembedding"),
            ("image_generation", "aimage_generation"),
            ("transcription", "atranscription"),
            ("speech", "aspeech"),
            ("rerank", "arerank"),
        ]
        for sync, async_ in pairs:
            assert OpenTelemetry._map_call_type_to_operation_name(
                sync
            ) == OpenTelemetry._map_call_type_to_operation_name(async_), (
                f"Sync '{sync}' and async '{async_}' should map to the same operation name"
            )


class TestRecordMetricsOperationName(unittest.TestCase):
    """Tests that _record_metrics passes the correct gen_ai.operation.name to histograms."""

    def _make_otel_with_mock_histogram(self):
        otel = OpenTelemetry()
        mock_histogram = MagicMock()
        otel._operation_duration_histogram = mock_histogram
        otel._token_usage_histogram = None
        otel._cost_histogram = None
        otel._time_to_first_token_histogram = None
        otel._time_per_output_token_histogram = None
        otel._response_duration_histogram = None
        return otel, mock_histogram

    def _call_record_metrics(self, otel, call_type, model="gpt-4", provider="openai"):
        from datetime import datetime, timedelta

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = start + timedelta(seconds=1)
        kwargs = {
            "model": model,
            "litellm_params": {"custom_llm_provider": provider},
            "standard_logging_object": {"call_type": call_type, "metadata": {}},
        }
        response_obj = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        otel._record_metrics(kwargs, response_obj, start, end)

    def test_completion_uses_chat_label(self):
        otel, mock_histogram = self._make_otel_with_mock_histogram()
        self._call_record_metrics(otel, "completion")
        attrs = mock_histogram.record.call_args[1]["attributes"]
        assert attrs["gen_ai.operation.name"] == "chat"

    def test_acompletion_uses_chat_label(self):
        otel, mock_histogram = self._make_otel_with_mock_histogram()
        self._call_record_metrics(otel, "acompletion")
        attrs = mock_histogram.record.call_args[1]["attributes"]
        assert attrs["gen_ai.operation.name"] == "chat"

    def test_embedding_uses_embeddings_label(self):
        """Regression: was hardcoded to 'chat' before this fix."""
        otel, mock_histogram = self._make_otel_with_mock_histogram()
        self._call_record_metrics(otel, "embedding", model="text-embedding-ada-002")
        attrs = mock_histogram.record.call_args[1]["attributes"]
        assert attrs["gen_ai.operation.name"] == "embeddings", (
            f"Expected 'embeddings', got {attrs['gen_ai.operation.name']!r}"
        )

    def test_transcription_uses_transcription_label(self):
        otel, mock_histogram = self._make_otel_with_mock_histogram()
        self._call_record_metrics(otel, "transcription", model="whisper-1")
        attrs = mock_histogram.record.call_args[1]["attributes"]
        assert attrs["gen_ai.operation.name"] == "transcription"

    def test_rerank_uses_rerank_label(self):
        otel, mock_histogram = self._make_otel_with_mock_histogram()
        self._call_record_metrics(otel, "rerank", model="rerank-english-v2.0", provider="cohere")
        attrs = mock_histogram.record.call_args[1]["attributes"]
        assert attrs["gen_ai.operation.name"] == "rerank"

    def test_missing_call_type_defaults_to_chat(self):
        """If standard_logging_object is absent, fall back to 'completion' -> 'chat'."""
        otel, mock_histogram = self._make_otel_with_mock_histogram()
        from datetime import datetime, timedelta

        start = datetime(2024, 1, 1)
        end = start + timedelta(seconds=1)
        kwargs = {
            "model": "gpt-4",
            "litellm_params": {"custom_llm_provider": "openai"},
        }
        otel._record_metrics(kwargs, {}, start, end)
        attrs = mock_histogram.record.call_args[1]["attributes"]
        assert attrs["gen_ai.operation.name"] == "chat"


class TestSetAttributesMessageLoggingGuards(unittest.TestCase):
    """
    Tests that gen_ai.operation.name and gen_ai.response.finish_reasons are
    emitted regardless of message-logging settings.

    Regression tests for: fix(otel): emit operation_name and finish_reasons
    regardless of message logging.
    """

    def _make_kwargs(self, call_type="completion"):
        return {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": call_type,
                "metadata": {},
            },
        }

    def _make_response_obj(self, finish_reason="stop"):
        return {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": "Hi"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    @patch("litellm.turn_off_message_logging", True)
    def test_operation_name_emitted_when_turn_off_message_logging_true(self):
        """gen_ai.operation.name must be set even when turn_off_message_logging=True."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        otel.set_attributes(
            span=mock_span,
            kwargs=self._make_kwargs(),
            response_obj=self._make_response_obj(),
        )

        set_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertIn("gen_ai.operation.name", set_keys)

        # Confirm the value is correct
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "chat")

    @patch("litellm.turn_off_message_logging", True)
    def test_finish_reasons_emitted_when_turn_off_message_logging_true(self):
        """gen_ai.response.finish_reasons must be set even when turn_off_message_logging=True."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        otel.set_attributes(
            span=mock_span,
            kwargs=self._make_kwargs(),
            response_obj=self._make_response_obj(finish_reason="stop"),
        )

        finish_calls = [
            call for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "gen_ai.response.finish_reasons"
        ]
        self.assertEqual(len(finish_calls), 1, "Should have exactly one gen_ai.response.finish_reasons attribute")
        self.assertEqual(json.loads(finish_calls[0][0][1]), ["stop"])

    @patch("litellm.turn_off_message_logging", True)
    def test_input_messages_not_emitted_when_turn_off_message_logging_true(self):
        """gen_ai.input.messages must NOT be set when turn_off_message_logging=True (logging guard still works)."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        otel.set_attributes(
            span=mock_span,
            kwargs=self._make_kwargs(),
            response_obj=self._make_response_obj(),
        )

        set_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn("gen_ai.input.messages", set_keys)
        self.assertNotIn("gen_ai.output.messages", set_keys)

    def test_operation_name_emitted_when_message_logging_false(self):
        """gen_ai.operation.name must be set even when otel.message_logging=False."""
        otel = OpenTelemetry()
        otel.message_logging = False
        mock_span = MagicMock()

        otel.set_attributes(
            span=mock_span,
            kwargs=self._make_kwargs(),
            response_obj=self._make_response_obj(),
        )

        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "chat")

    def test_finish_reasons_emitted_when_message_logging_false(self):
        """gen_ai.response.finish_reasons must be set even when otel.message_logging=False."""
        otel = OpenTelemetry()
        otel.message_logging = False
        mock_span = MagicMock()

        otel.set_attributes(
            span=mock_span,
            kwargs=self._make_kwargs(),
            response_obj=self._make_response_obj(finish_reason="length"),
        )

        finish_calls = [
            call for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "gen_ai.response.finish_reasons"
        ]
        self.assertEqual(len(finish_calls), 1)
        self.assertEqual(json.loads(finish_calls[0][0][1]), ["length"])

    @patch("litellm.turn_off_message_logging", True)
    def test_operation_name_uses_call_type_for_non_completion(self):
        """gen_ai.operation.name reflects call_type for non-completion calls, even with logging off."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        otel.set_attributes(
            span=mock_span,
            kwargs=self._make_kwargs(call_type="embedding"),
            response_obj={"id": "r", "model": "text-embedding-ada-002", "choices": [], "usage": {}},
        )

        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "embeddings")

    def test_finish_reasons_absent_when_no_finish_reason_in_choices(self):
        """gen_ai.response.finish_reasons must not be set when all choices lack a finish_reason."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        response_obj = {
            "id": "r",
            "model": "gpt-4",
            "choices": [{"finish_reason": None, "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        }

        otel.set_attributes(span=mock_span, kwargs=self._make_kwargs(), response_obj=response_obj)

        set_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn("gen_ai.response.finish_reasons", set_keys)

    def test_finish_reasons_absent_when_response_has_no_choices(self):
        """gen_ai.response.finish_reasons must not be set when choices list is empty."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        response_obj = {
            "id": "r",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        }

        otel.set_attributes(span=mock_span, kwargs=self._make_kwargs(), response_obj=response_obj)

        set_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn("gen_ai.response.finish_reasons", set_keys)


class TestGetSemconvMode(unittest.TestCase):
    """Tests for _get_semconv_mode() — the env-var-driven semconv selector."""

    def test_default_returns_old(self):
        """Absent env var → old mode (backward-compatible default)."""
        from litellm.integrations.opentelemetry import _get_semconv_mode, _SEMCONV_MODE_OLD

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
            self.assertEqual(_get_semconv_mode(), _SEMCONV_MODE_OLD)

    def test_gen_ai_latest_experimental_returns_latest(self):
        """Setting OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental → latest mode."""
        from litellm.integrations.opentelemetry import _get_semconv_mode, _SEMCONV_MODE_LATEST

        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            self.assertEqual(_get_semconv_mode(), _SEMCONV_MODE_LATEST)

    def test_unrecognized_value_falls_back_to_old(self):
        """An unrecognized value should not enable latest mode."""
        from litellm.integrations.opentelemetry import _get_semconv_mode, _SEMCONV_MODE_OLD

        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "some_other_value"}):
            self.assertEqual(_get_semconv_mode(), _SEMCONV_MODE_OLD)

    def test_comma_separated_including_latest_returns_latest(self):
        """Comma-separated list that includes gen_ai_latest_experimental → latest mode."""
        from litellm.integrations.opentelemetry import _get_semconv_mode, _SEMCONV_MODE_LATEST

        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "http, gen_ai_latest_experimental"}):
            self.assertEqual(_get_semconv_mode(), _SEMCONV_MODE_LATEST)


class TestNormalizeProviderName(unittest.TestCase):
    """Tests for OpenTelemetry._normalize_provider_name() static method."""

    def _norm(self, provider: str) -> str:
        return OpenTelemetry._normalize_provider_name(provider)

    def test_known_providers_mapped_correctly(self):
        cases = [
            ("openai",        "openai"),
            ("azure",         "azure.ai.openai"),
            ("azure_ai",      "azure.ai.openai"),
            ("anthropic",     "anthropic"),
            ("bedrock",       "aws.bedrock"),
            ("vertex_ai",     "gcp.vertex_ai"),
            ("vertex_ai_beta","gcp.vertex_ai"),
            ("gemini",        "gcp.gemini"),
            ("cohere",        "cohere"),
            ("cohere_chat",   "cohere"),
            ("deepseek",      "deepseek"),
            ("groq",          "groq"),
            ("mistral",       "mistral_ai"),
            ("perplexity",    "perplexity"),
            ("xai",           "x_ai"),
        ]
        for provider, expected in cases:
            with self.subTest(provider=provider):
                self.assertEqual(self._norm(provider), expected)

    def test_unknown_provider_is_returned_as_is(self):
        """An unmapped provider string should pass through unchanged."""
        self.assertEqual(self._norm("mycustomprovider"), "mycustomprovider")

    def test_normalization_is_case_insensitive(self):
        """Provider names should be matched case-insensitively."""
        self.assertEqual(self._norm("OpenAI"),    "openai")
        self.assertEqual(self._norm("ANTHROPIC"), "anthropic")
        self.assertEqual(self._norm("Bedrock"),   "aws.bedrock")


class TestSpanKindSemconv(unittest.TestCase):
    """Tests that span kind=CLIENT is set in new semconv mode and absent in old mode."""

    _BASE_KWARGS = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
        "standard_logging_object": {
            "id": "req-1",
            "call_type": "completion",
            "metadata": {},
            "request_id": "req-1",
        },
    }

    def _make_otel(self, env: dict) -> OpenTelemetry:
        with patch.dict(os.environ, env):
            otel = OpenTelemetry()
        return otel

    def test_new_semconv_sets_span_kind_client(self):
        """In latest semconv mode, start_span must be called with kind=SpanKind.CLIENT."""
        from opentelemetry.trace import SpanKind

        otel = self._make_otel({"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"})
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        otel.get_tracer_to_use_for_request = MagicMock(return_value=mock_tracer)

        import datetime
        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=1)
        otel._handle_success(self._BASE_KWARGS.copy(), MagicMock(), start, end)

        # The inference span is the first start_span call; raw request span is second.
        first_call_kwargs = mock_tracer.start_span.call_args_list[0][1]
        self.assertIn("kind", first_call_kwargs)
        self.assertEqual(first_call_kwargs["kind"], SpanKind.CLIENT)

    def test_old_semconv_does_not_set_span_kind(self):
        """In default (old) semconv mode, start_span must NOT include a kind kwarg."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
            otel = OpenTelemetry()

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        otel.get_tracer_to_use_for_request = MagicMock(return_value=mock_tracer)

        import datetime
        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=1)
        otel._handle_success(self._BASE_KWARGS.copy(), MagicMock(), start, end)

        # Verify that no start_span call (inference or raw) was given a kind kwarg.
        for call in mock_tracer.start_span.call_args_list:
            self.assertNotIn("kind", call[1])


class TestSpanNameSemconv(unittest.TestCase):
    """Tests for _get_span_name() under different semconv modes."""

    def _make_kwargs(self, model="gpt-4o", call_type="completion", generation_name=None):
        metadata = {}
        if generation_name:
            metadata["generation_name"] = generation_name
        return {
            "model": model,
            "litellm_params": {"custom_llm_provider": "openai", "metadata": metadata},
            "standard_logging_object": {"call_type": call_type},
        }

    def test_new_semconv_span_name_is_operation_space_model(self):
        """Latest mode: span name should be '{operation} {model}', e.g. 'chat gpt-4o'."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        name = otel._get_span_name(self._make_kwargs(model="gpt-4o", call_type="completion"))
        self.assertEqual(name, "chat gpt-4o")

    def test_new_semconv_span_name_uses_correct_operation_for_embedding(self):
        """Latest mode: embedding call_type → 'embeddings {model}'."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        name = otel._get_span_name(self._make_kwargs(model="text-embedding-3-small", call_type="embedding"))
        self.assertEqual(name, "embeddings text-embedding-3-small")

    def test_generation_name_override_takes_priority_in_new_semconv(self):
        """generation_name in metadata always wins over the new semconv format."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        name = otel._get_span_name(self._make_kwargs(generation_name="my-custom-span"))
        self.assertEqual(name, "my-custom-span")

    def test_old_semconv_span_name_is_litellm_request(self):
        """Default (old) mode: span name is LITELLM_REQUEST_SPAN_NAME (backward compat)."""
        from litellm.integrations.opentelemetry import LITELLM_REQUEST_SPAN_NAME

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
            otel = OpenTelemetry()

        name = otel._get_span_name(self._make_kwargs())
        self.assertEqual(name, LITELLM_REQUEST_SPAN_NAME)


class TestProviderAttributeSemconv(unittest.TestCase):
    """Tests that the correct provider attribute is set based on semconv mode."""

    _BASE_KWARGS = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
        "standard_logging_object": {
            "id": "req-1",
            "call_type": "completion",
            "metadata": {},
        },
    }

    def _call_set_attributes(self, otel: OpenTelemetry) -> MagicMock:
        mock_span = MagicMock()
        response_obj = {
            "id": "resp-1",
            "model": "gpt-4o",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        otel.set_attributes(span=mock_span, kwargs=self._BASE_KWARGS.copy(), response_obj=response_obj)
        return mock_span

    def _set_attribute_keys(self, mock_span: MagicMock):
        return [call[0][0] for call in mock_span.set_attribute.call_args_list]

    def test_old_semconv_sets_gen_ai_system_not_provider_name(self):
        """Old mode: gen_ai.system is set; gen_ai.provider.name is NOT set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
            otel = OpenTelemetry()

        mock_span = self._call_set_attributes(otel)
        keys = self._set_attribute_keys(mock_span)

        self.assertIn("gen_ai.system", keys)
        self.assertNotIn("gen_ai.provider.name", keys)

    def test_new_semconv_sets_gen_ai_provider_name_not_gen_ai_system(self):
        """New mode: gen_ai.provider.name is set (normalized); gen_ai.system is NOT set."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        mock_span = self._call_set_attributes(otel)
        keys = self._set_attribute_keys(mock_span)

        self.assertIn("gen_ai.provider.name", keys)
        self.assertNotIn("gen_ai.system", keys)

    def test_new_semconv_provider_name_value_is_normalized(self):
        """New mode: the gen_ai.provider.name value is the canonical normalized form."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        mock_span = self._call_set_attributes(otel)
        provider_calls = [
            call[0][1]
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "gen_ai.provider.name"
        ]
        self.assertEqual(len(provider_calls), 1)
        self.assertEqual(provider_calls[0], "openai")


class TestTokenAttributesSemconv(unittest.TestCase):
    """Tests that the correct token attributes are set based on semconv mode."""

    _BASE_KWARGS = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
        "standard_logging_object": {
            "id": "req-1",
            "call_type": "completion",
            "metadata": {},
        },
    }
    _RESPONSE = {
        "id": "resp-1",
        "model": "gpt-4o",
        "choices": [],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    def _set_attrs(self, otel: OpenTelemetry) -> MagicMock:
        mock_span = MagicMock()
        otel.set_attributes(span=mock_span, kwargs=self._BASE_KWARGS.copy(), response_obj=self._RESPONSE)
        return mock_span

    def _attr_keys(self, mock_span: MagicMock):
        return [call[0][0] for call in mock_span.set_attribute.call_args_list]

    def test_old_semconv_emits_both_legacy_and_new_token_names(self):
        """Old mode emits legacy (llm.usage.*) names plus the v1.38 gen_ai.usage.* names."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
            otel = OpenTelemetry()

        mock_span = self._set_attrs(otel)
        keys = self._attr_keys(mock_span)

        # Legacy OpenLLMetry names
        self.assertIn("gen_ai.usage.prompt_tokens",     keys)
        self.assertIn("gen_ai.usage.completion_tokens", keys)
        # New v1.38 names
        self.assertIn("gen_ai.usage.input_tokens",  keys)
        self.assertIn("gen_ai.usage.output_tokens", keys)
        self.assertIn("gen_ai.usage.total_tokens",  keys)

    def test_new_semconv_emits_only_new_token_names(self):
        """New mode emits only gen_ai.usage.input/output/total_tokens; no legacy names."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        mock_span = self._set_attrs(otel)
        keys = self._attr_keys(mock_span)

        # New names present
        self.assertIn("gen_ai.usage.input_tokens",  keys)
        self.assertIn("gen_ai.usage.output_tokens", keys)
        self.assertIn("gen_ai.usage.total_tokens",  keys)
        # Legacy names absent
        self.assertNotIn("gen_ai.usage.prompt_tokens",     keys)
        self.assertNotIn("gen_ai.usage.completion_tokens", keys)

    def test_new_semconv_token_values_are_correct(self):
        """New mode token attribute values match usage from the response."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        mock_span = self._set_attrs(otel)
        token_values = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

        self.assertEqual(token_values.get("gen_ai.usage.input_tokens"),  100)
        self.assertEqual(token_values.get("gen_ai.usage.output_tokens"),  50)
        self.assertEqual(token_values.get("gen_ai.usage.total_tokens"),  150)


class TestToolDefinitionsSemconv(unittest.TestCase):
    """Tests that tool definitions are emitted in the correct format per semconv mode."""

    _TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                # No description, no parameters
            },
        },
    ]

    def _call(self, otel: OpenTelemetry) -> MagicMock:
        mock_span = MagicMock()
        otel.set_tools_attributes(span=mock_span, tools=self._TOOLS)
        return mock_span

    def _attr(self, mock_span: MagicMock) -> dict:
        return {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    def test_old_semconv_uses_indexed_flat_attributes(self):
        """Old mode: tools are stored as llm.request.functions.{i}.name/description/parameters."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
            otel = OpenTelemetry()

        attrs = self._attr(self._call(otel))

        self.assertIn("llm.request.functions.0.name",        attrs)
        self.assertIn("llm.request.functions.0.description", attrs)
        self.assertIn("llm.request.functions.0.parameters",  attrs)
        self.assertIn("llm.request.functions.1.name",        attrs)
        self.assertNotIn("gen_ai.tool.definitions",           attrs)

    def test_new_semconv_uses_gen_ai_tool_definitions_json_array(self):
        """New mode: tools are stored as a single gen_ai.tool.definitions JSON array."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        attrs = self._attr(self._call(otel))

        self.assertIn("gen_ai.tool.definitions", attrs)
        parsed = json.loads(attrs["gen_ai.tool.definitions"])
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["name"], "get_weather")
        self.assertEqual(parsed[0]["description"], "Get the current weather")
        self.assertIn("parameters", parsed[0])
        # No indexed flat attributes
        self.assertNotIn("llm.request.functions.0.name", attrs)

    def test_new_semconv_omits_missing_description_key(self):
        """New mode: tool with no description should not have description key in the JSON object."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        attrs = self._attr(self._call(otel))
        parsed = json.loads(attrs["gen_ai.tool.definitions"])
        no_desc_tool = next(t for t in parsed if t["name"] == "get_time")
        self.assertNotIn("description", no_desc_tool)


class TestRequestIdAlwaysEmitted(unittest.TestCase):
    """
    Tests that gen_ai.request.id is recorded regardless of message-logging settings.
    Regression guard for the move of the request_id block before the logging guard.
    """

    _BASE_KWARGS = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
        "standard_logging_object": {
            "id": "req-xyz",
            "call_type": "completion",
            "metadata": {},
            "request_id": "req-xyz",
        },
    }
    _RESPONSE = {
        "id": "resp-1",
        "model": "gpt-4o",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    def _call_set_attributes(self, otel: OpenTelemetry) -> MagicMock:
        mock_span = MagicMock()
        otel.set_attributes(span=mock_span, kwargs=self._BASE_KWARGS.copy(), response_obj=self._RESPONSE)
        return mock_span

    def test_request_id_emitted_when_turn_off_message_logging_true(self):
        """gen_ai.request.id must be set even when turn_off_message_logging=True."""
        import litellm

        otel = OpenTelemetry()
        original = litellm.turn_off_message_logging
        try:
            litellm.turn_off_message_logging = True
            mock_span = self._call_set_attributes(otel)
        finally:
            litellm.turn_off_message_logging = original

        keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertIn("gen_ai.request.id", keys)

    def test_request_id_emitted_when_message_logging_disabled(self):
        """gen_ai.request.id must be set even when otel.message_logging=False."""
        otel = OpenTelemetry()
        otel.message_logging = False

        mock_span = self._call_set_attributes(otel)
        keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertIn("gen_ai.request.id", keys)


class TestMetricsProviderAttributeSemconv(unittest.TestCase):
    """Tests that _record_metrics uses the correct provider attribute per semconv mode."""

    _BASE_KWARGS = {
        "model": "gpt-4o",
        "litellm_params": {"custom_llm_provider": "anthropic"},
        "standard_logging_object": {
            "call_type": "completion",
            "metadata": {},
        },
    }
    _RESPONSE = {
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    def _call_record_metrics(self, otel: OpenTelemetry):
        """Call _record_metrics with a mocked histogram and return the attributes dict."""
        import datetime

        mock_histogram = MagicMock()
        otel._operation_duration_histogram = mock_histogram

        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=1)
        otel._record_metrics(self._BASE_KWARGS.copy(), self._RESPONSE, start, end)

        # The first positional arg is the duration; attributes are the keyword arg
        call_kwargs = mock_histogram.record.call_args[1]
        return call_kwargs["attributes"]

    def test_old_semconv_metrics_use_gen_ai_system(self):
        """Old mode: common_attrs includes gen_ai.system; gen_ai.provider.name absent."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
            otel = OpenTelemetry()

        attrs = self._call_record_metrics(otel)

        self.assertIn("gen_ai.system", attrs)
        self.assertEqual(attrs["gen_ai.system"], "anthropic")
        self.assertNotIn("gen_ai.provider.name", attrs)

    def test_new_semconv_metrics_use_gen_ai_provider_name(self):
        """New mode: common_attrs includes gen_ai.provider.name (normalized); gen_ai.system absent."""
        with patch.dict(os.environ, {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
            otel = OpenTelemetry()

        attrs = self._call_record_metrics(otel)

        self.assertIn("gen_ai.provider.name", attrs)
        self.assertEqual(attrs["gen_ai.provider.name"], "anthropic")
        self.assertNotIn("gen_ai.system", attrs)
