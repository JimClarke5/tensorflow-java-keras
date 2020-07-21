/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.initializers;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;

/**
 * Test the InitializerFunction initializer
 */
public class InitializerFunctionTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public InitializerFunctionTest() {
    }

    @BeforeAll
    public static void setUpClass() {
    }

    @AfterAll
    public static void tearDownClass() {
    }

    @BeforeEach
    public void setUp() {
    }

    @AfterEach
    public void tearDown() {
    }

    /**
     * Test of call method, of class OptimizerFunction.
     */
    @Test
    public void testLambdaCallFloat() {
        float[] expected = {12345.0f};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TFloat32> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(new float[]{12345.0f}), dtype);
            };
            Operand<TFloat32> operand = func.call(tf.constant(shape), TFloat32.DTYPE);
            session.evaluate(expected, operand);

        }
    }

    /**
     * Test of call method, of class OptimizerFunction.
     */
    @Test
    public void testLambdaCallUInt8() {
        byte[] expected = {0x15};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TUint8> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(new byte[]{0x15}), dtype);
            };
            Operand<TUint8> operand = func.call(tf.constant(shape), TUint8.DTYPE);
            session.evaluate(expected, operand);

        }
    }

    /**
     * Test of call method, of class OptimizerFunction.
     */
    @Test
    public void testLambdaCallInt() {
        int[] expected = {12345};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TInt32> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(new int[]{12345}), dtype);
            };
            Operand<TInt32> operand = func.call(tf.constant(shape), TInt32.DTYPE);
            session.evaluate(expected, operand);

        }
    }

    /**
     * Test of call method, of class OptimizerFunction.
     */
    @Test
    public void testLambdaCallLong() {
        long[] expected = {12345L};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TInt64> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(new long[]{12345L}), dtype);
            };
            Operand<TInt64> operand = func.call(tf.constant(shape), TInt64.DTYPE);
            session.evaluate(expected, operand);

        }
    }

    public void testLambdaCallDouble() {
        double[] expected = {Math.PI};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(1);

            // Test float
            InitializerFunction<TFloat64> func = (dims, dtype) -> {
                return tf.dtypes.cast(tf.constant(new double[]{Math.PI}), dtype);
            };
            Operand<TFloat64> operand = func.call(tf.constant(shape), TFloat64.DTYPE);
            session.evaluate(expected, operand);

        }
    }

}
