/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.backend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.losses.impl.LossesImpl;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ReduceSum;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.SoftmaxCrossEntropyWithLogits;
import org.tensorflow.op.nn.SparseSoftmaxCrossEntropyWithLogits;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TBfloat16;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat16;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TNumber;
import org.tensorflow.types.family.TType;

/**
 *
 * @author Jim Clarke
 */
public class K {
    public static final double Epsilon = 1e-7;
    public static final float EpsilonF = 1e-7F;
    
    public static final double epsilon() { return Epsilon; }
    
    public static final Operand epsilonConstant(Ops tf, DataType dtype) { return tf.dtypes.cast(tf.constant(Epsilon), dtype); }
    public static final Operand one(Ops tf, DataType dtype) { return tf.dtypes.cast(tf.constant(1), dtype); }
    public static final Operand zero(Ops tf, DataType dtype) { return tf.dtypes.cast(tf.constant(0), dtype); }
    public static final Operand constant(Ops tf, double number, DataType dtype) { return tf.dtypes.cast(tf.constant(number), dtype); }
    
    
    public static Operand clip(Ops tf, Operand x, double minValue, double maxValue) {
        assert x != null : "Operand x must not be null";
        DataType dtype = x.asOutput().dataType();
        if(maxValue < minValue) {
            maxValue = minValue;
        }
        Operand minValueConstant = tf.dtypes.cast(tf.constant(minValue), dtype);
        Operand maxValueConstant = tf.dtypes.cast(tf.constant(maxValue), dtype);
        return tf.clipByValue(x, minValueConstant, maxValueConstant);
    }
    
    public static Operand mean(Ops tf, Operand x) {
        return mean(tf, x, tf.constant(-1), false);
    }
    public static Operand mean(Ops tf, Operand x, Operand axis) {
        return mean(tf, x, axis, false);
    }
    
    public static Operand mean(Ops tf, Operand x,boolean keepDims) {
        return mean(tf, x, tf.constant(-1), keepDims);
    }
    
    public static Operand mean(Ops tf, Operand x, Operand axis, boolean keepDims) {
        if(x.asOutput().dataType() == TBool.DTYPE) {
            x = tf.dtypes.cast(x, TFloat32.DTYPE);
        }
        return tf.math.mean(x, axis, Mean.keepDims(keepDims));
    }
    
    public static Operand maximum(Ops tf, Operand x, Operand y) {
        y = tf.dtypes.cast(y, x.asOutput().dataType());
        return tf.math.maximum(x,y);
    }
    
    public static Shape merge(Shape a, Shape b) {
        assert a.numDimensions() == b.numDimensions() : String.format("Shapes %s and %s are incompatible", a, b);
        long[] array = new long[a.numDimensions()];
        for(int i = 0; i < a.numDimensions(); i++) {
            if(a.size(i) != Shape.UNKNOWN_SIZE) {
                if(b.size(i) != Shape.UNKNOWN_SIZE)
                    assert a.size(i) == b.size(i) : String.format("Shapes %s and %s are incompatible", a, b);
                array[i] = a.size(i);
            } else
                array[i] = b.size(i);
        }
        return Shape.of(array);
        
    }
    
    
    // this is from nn in Python, I could not find it in the Java frameworks.
    public static Operand sigmoidCrossEntropyWithLogits(Ops tf, Operand labels, Operand logits) {
        Shape lablesShape = labels.asOutput().shape();
        Shape logitsShape = logits.asOutput().shape();
        Shape newShape = merge(lablesShape, logitsShape);
        
        Operand zeros = tf.dtypes.cast(tf.zerosLike(logits), logits.asOutput().dataType());
        Operand cond = tf.math.greaterEqual(logits, zeros);
        
        Operand relu_logits = tf.select(cond, logits, zeros);
        Operand neg_abs_logits = tf.select(cond, tf.math.neg(logits), logits);
        return tf.math.add(
             tf.math.sub(relu_logits, tf.math.mul(logits, labels)),
             tf.math.log1p(tf.math.exp(neg_abs_logits))
        );
        
    }
    //TODO Not sure if this is right or how to do it.
    private static Operand  backtrackIdentity(Operand output) {
       // while(!output.op().type().equals("Identity"))
        //    output = output.op().output(0);
        return output;
    }
    
    public static Operand binary_crossentropy(Ops tf, Operand target, Operand output, boolean fromLogits ){
        if(fromLogits) {
            return sigmoidCrossEntropyWithLogits(tf, target, output);
        }
        
        if(!(output instanceof Variable) && (!tf.scope().env().isEager())) {
            //output = backtrackIdentity(output); // TODO - this does not work, goes infinite loop
            if(output.op().type().equals("Sigmoid")) {
                assert output.op().numOutputs() == 1;
                output = output.op().output(0);
                return sigmoidCrossEntropyWithLogits(tf, target, output);
            }
        }
        DataType dtype = output.asOutput().dataType();
        Operand one = one(tf,dtype);
        Operand epsilonConst = K.epsilonConstant(tf,dtype);
        Operand oneMinusEpsilonConst = tf.math.sub(one, epsilonConst);
        output = tf.clipByValue(output, epsilonConst, oneMinusEpsilonConst);
        
        // Compute cross entropy from probabilities.
        Operand bce = tf.math.mul(target, tf.math.log(tf.math.add(output, epsilonConst)));
        bce = tf.math.add(bce,
            tf.math.mul(
                tf.math.sub(one, target),
                tf.math.log(tf.math.add(tf.math.sub(one, output), epsilonConst )) 
            ));
        Operand result =  tf.math.neg(bce);
        return result;
    }
    
    public static Operand categorical_crossentropy(Ops tf, Operand target, Operand output, boolean fromLogits) {
        return categorical_crossentropy(tf, target, output, fromLogits, -1);
    }

    public static Operand categorical_crossentropy(Ops tf, Operand target, Operand output, boolean fromLogits, int axis) {
        
        if(fromLogits) {
            return softmax_cross_entropy_with_logits(tf, target, output);
        }
        if(!(output instanceof Variable) && (!tf.scope().env().isEager())) {
            //TODO output = backtrackIdentity(output); doesn't seem to work with Java version.
            if(output.op().type().equals("Softmax")) {
                assert output.op().numOutputs() == 1;
                output = output.op().output(0);
                 Operand op = softmax_cross_entropy_with_logits(tf, target, output);
                 return op;
            }
        }
        DataType dtype = output.asOutput().dataType();
        Operand one = one(tf,dtype);
        Operand epsilonConst = K.epsilonConstant(tf,dtype);
        Operand oneMinusepsilonConst = tf.math.sub(one, epsilonConst);
        output = tf.math.div(output,  tf.reduceSum(output, tf.constant(axis), ReduceSum.keepDims(Boolean.TRUE)));
        output = tf.clipByValue(output, epsilonConst, oneMinusepsilonConst);
        
        // Compute cross entropy from probabilities.
        Operand cce = tf.reduceSum(tf.math.mul(target, tf.math.log(output)), 
                tf.constant(axis), ReduceSum.keepDims(Boolean.FALSE));
        return tf.math.neg(cce);
    }
    
    public static Operand flatten(Ops tf, Operand t) {
        Shape shape = Shape.of(1L);
        return tf.reshape(t, tf.constant(shape));
        
    }
    
    
    public static Operand sparse_categorical_crossentropy(Ops tf, Operand target, Operand output, boolean fromLogits, int axis) {
        DataType dType = output.asOutput().dataType();
        if(!(output instanceof Variable) && (!tf.scope().env().isEager())) {
            //TODO output = backtrackIdentity(output); doesn't seem to work with Java version.
            if(output.op().type().equals("Softmax")) {
                //assert output.op().numOutputs() == 1;
                // When softmax activation function is used for output operation, we
                // use logits from the softmax function directly to compute loss in order
                // to prevent collapsing zero when training.
                //TODO assert len(output.op.inputs) == 1
                //TODO output = output.op.inputs[0]
                fromLogits = true;
            }
        }
        if(!fromLogits) {
            Operand epsilonConst = epsilonConstant(tf,dType);
            Operand one = one(tf, dType);
            Operand oneMinusEpsilonConst = tf.math.sub(one, epsilonConst);
            output = tf.clipByValue(output, epsilonConst, oneMinusEpsilonConst);
            output = tf.math.log(output);
        }
        Shape outputShape = output.asOutput().shape();
        int outputRank = outputShape.numDimensions();
        axis %= outputRank;
        if(axis < 0)
            axis += outputRank;
        if(axis != outputRank - 1){
            //TODO permutation = list(
            //TODO itertools.chain(range(axis), range(axis + 1, output_rank), [axis]))

            int[] axisNew = moveAxisToEnd(axis, outputRank);
            List<Operand<TInt32>> permutationList = new ArrayList<>();
            for(int i : axisNew) {
                permutationList.add(tf.constant(i));
            }
            output =  tf.linalg.transpose( output, 
                tf.concat( permutationList, tf.constant(0))
            );
        }
        
        target = tf.dtypes.cast(target, TInt64.DTYPE);
        // TODO Try to adjust the shape so that rank of labels = rank of logits - 1.
        outputShape = output.asOutput().shape();
        Shape targetShape = target.asOutput().shape();
        int targetRank = targetShape.numDimensions();
        
        boolean updateShape = targetRank != outputRank - 1;
        if(updateShape) { // TODO check to see if this is right
            target = tf.reshape(target, tf.constant(-1L)); // flatten
            output = tf.reshape(output, tf.constant(new long[]{-1L, outputShape.size(outputShape.numDimensions()-1)}));
        }

        // call nn.nn.sparse_softmax_cross_entropy_with_logits_v2
        Operand loss = sparse_softmax_cross_entropy_with_logits(tf, target, output);
        if (updateShape && outputRank >= 3) {
            long[] dims = outputShape.asArray();
            long[] newDims = new long[dims.length-1];
            System.arraycopy(dims, 0, newDims, 0, newDims.length);
            loss = tf.reshape(loss, tf.constant(newDims));
        }
        return loss;
    }
    
   
    
    private static int[] allAxis(Operand op) {
        int rank = op.asOutput().shape().numDimensions();
        int[] ranks = new int[rank];
        for(int i = 0; i < rank; i++)
            ranks[i] = i;
        return ranks;
    }
    
    public static Operand allAxis(Ops tf, Operand op) {
        int[] ranks = allAxis(op);
        return tf.constant(ranks);
    }
    
    //TODO shouldn't these be in tensorflow itself under nn?
    private static <T extends TType,U extends TNumber> Operand moveDimToEnd(Ops tf, Operand tensor, int dim_index, Operand rank){
        Operand one = one(tf, TInt32.DTYPE);
        List<Operand<T>> concatList = Arrays.asList(
                tf.range(tf.constant(dim_index), one, one),
                tf.range(tf.constant(dim_index+1),rank, one)
        );
        return tf.linalg.transpose( tensor, 
                (Operand<U>)tf.concat( 
                        (Iterable<Operand<T>>)concatList, 
                        (Operand<U>)tf.constant(0)));
    }
    
    private static <T extends TType,U extends TNumber>Operand flattenOuterDims(Ops tf, Operand logits) {
        Operand zero = zero(tf, TInt64.DTYPE);
        Operand one = one(tf, TInt64.DTYPE);
        Operand minusOne = tf.constant(-1);
        
        //Shape logitsShape = logits.asOutput().shape();
        //long lastDimSize = logitsShape.size(logitsShape.numDimensions()-1);
        //if(!tf.scope().env().isEager()) {
            Shape shape = logits.asOutput().shape();
            int ndims = shape.numDimensions();
            if(!shape.hasUnknownDimension()) {
                long product = 1L;
                boolean productValid = true;
                for(int i = ndims-2; i >= 0; i--) {
                    long d = shape.size(i);
                    if(d == Shape.UNKNOWN_SIZE) {
                        productValid = false;
                        break;
                    }
                    product *= d;
                }
                if(productValid) {
                    Shape outputShape = Shape.of(product, shape.size(ndims-1));
                    return tf.reshape(logits, tf.constant(outputShape.asArray()));
                }
            }
        //}
        
        Operand rank = tf.dtypes.cast(tf.rank(logits),TInt64.DTYPE);
        Operand rankMinusOne = tf.math.sub(rank, one);

        Operand last_dim_size = tf.slice(
                tf.shape(logits), 
                rankMinusOne,
                tf.constant(1)
           );
        Operand concat = tf.concat(Arrays.asList(tf.constant(new int[] {-1}), last_dim_size), tf.constant(0));
        return tf.reshape(zero, concat);
        
    }
    
    private static int[] moveAxisToEnd(int axis, int outputRank) {
        int[] axisNew = new int[outputRank];
         for(int i = 0; i < axis; i++) {
             axisNew[i] = i;
         }
         for(int i = axis+1; i < outputRank; i++) {
             axisNew[i-1] = i;
         }
         axisNew[outputRank-1] = axis;
         return axisNew;
            
    }
    
    // TODO, maybe part of Shape
    private static boolean shapeIsCompatible(Shape a, Shape b) {
        if(a.numDimensions() != b.numDimensions()) {
            return false;
        }
        for(int i = 0; i < a.numDimensions();i++) {
            long aSize = a.size(i);
            long bSize = b.size(i);
            if( aSize != Shape.UNKNOWN_SIZE &&
                    bSize != Shape.UNKNOWN_SIZE &&
                    aSize != bSize)
                return false;
        }
        return true;
    }
    
    // TODO these are "nn" ops
    public static Operand softmax_cross_entropy_with_logits(Ops tf, Operand labels, Operand logits) {
        return softmax_cross_entropy_with_logits(tf, labels, logits, -1);
    }
    public static Operand softmax_cross_entropy_with_logits(Ops tf, Operand labels, Operand logits, int axis) {
        axis = axis % logits.asOutput().shape().numDimensions();
        if(axis < 0)
            axis += logits.asOutput().shape().numDimensions();
                
        Operand minusOne = tf.constant(-1);
        Operand precise_logits = logits;
        Operand one = tf.constant(1L);
        
        boolean convertToFloat32 = logits.asOutput().dataType() == TFloat16.DTYPE ||
                logits.asOutput().dataType() == TBfloat16.DTYPE;
        if(convertToFloat32 )
            precise_logits = tf.dtypes.cast(logits, TFloat32.DTYPE);
        DataType dtype = precise_logits.asOutput().dataType();
        labels = tf.dtypes.cast(labels, dtype);
        Operand inputRank = tf.dtypes.cast(tf.rank(precise_logits), TInt64.DTYPE);
        Operand inputRankMinusOne = tf.dtypes.cast(tf.math.sub(inputRank, one), TInt64.DTYPE);
        Shape shape = logits.asOutput().shape();
        
        // Move the dim to the end if dim is not the last dimension.
        if(axis != -1 && axis != precise_logits.asOutput().shape().numDimensions() -1 ) {
          precise_logits = moveDimToEnd(tf, precise_logits, axis, inputRank);
          labels = moveDimToEnd(tf, labels, axis, inputRank);
        }

        Shape inputShape = precise_logits.asOutput().shape();
        precise_logits = flattenOuterDims(tf, precise_logits);
        labels = flattenOuterDims(tf, labels);
        SoftmaxCrossEntropyWithLogits smax = tf.nn.softmaxCrossEntropyWithLogits(
                precise_logits,  labels);
        Operand cost = smax.loss();
        Operand outputShape = tf.slice(tf.constant(inputShape.asArray()), 
                tf.constant(new long[]{0}), 
                tf.constant(new long[]{inputShape.numDimensions()-1}));
        cost = tf.reshape(cost,outputShape);
        if(tf.scope().env().isGraph() && !shape.hasUnknownDimension()) {
            long[] array = shape.asArray();
            long[] newArray = new long[array.length-1];
            if(axis < 0)
                axis = shape.numDimensions() + axis;
            for(int i = 0; i < axis; i++) {
                newArray[i] = shape.size(i);
            }
            for(int i = axis + 1; i < shape.numDimensions(); i++) {
                newArray[i-1] = shape.size(i);
            }
            Shape newShape = Shape.of(newArray);
            cost = tf.reshape(cost, tf.constant(newShape.asArray()));
        }
        
        if(convertToFloat32) 
            cost = tf.dtypes.cast(cost, logits.asOutput().dataType());
        return cost;
    }
    
    
    //TODO SHould these be in Shape, thes implemenations are basic and not totally robust.
    // like head but is not limited to first dimension
    public static Shape head(Shape oldShape, int axis) {
        axis %= oldShape.numDimensions();
        if(axis < 0)
            axis += oldShape.numDimensions();
        long[] array = oldShape.asArray();
        long[] newArray = new long[axis];
        for(int i = 0; i < axis; i++) {
            newArray[i] = array[i];
        }
        return Shape.of(newArray);
    }
    public static Shape tail(Shape oldShape, int axis) {
        axis %= oldShape.numDimensions();
        if(axis < 0)
            axis += oldShape.numDimensions();
        long[] array = oldShape.asArray();
        long[] newArray = new long[oldShape.numDimensions()- axis];
        for(int i = axis, j = 0; i < array.length; i++, j++) {
            newArray[j] = array[i];
        }
        return Shape.of(newArray);
    }
    
    /**
     * Reshapes the shape by eliminating trailing Dimensions.
     * @param oldShape
     * @param axis
     * @return the new shape
     */
    public static Shape squeeze(Shape oldShape, int axis) {
        axis %= oldShape.numDimensions();
        if(axis < 0)
            axis = oldShape.numDimensions() + axis;
        long[] array = oldShape.asArray();
        long[] newArray = new long[axis];
        for(int i = 0; i < axis-1; i++) {
            newArray[i] = array[i];
        }
        long sum = array[axis-1];
        for(int i = axis; i < array.length; i++) {
            sum *= array[i];
        }
        newArray[axis-1] = sum;
        return Shape.of(newArray);
    }
    
    public static Shape shorten(Shape oldShape, int count) {
        
        count %= oldShape.numDimensions();
        if(count < 0)
            count += oldShape.numDimensions();
        long[] array = oldShape.asArray();
        long[] newArray = new long[array.length - count];
        System.arraycopy(array, 0, newArray, 0, count);
        return Shape.of(newArray);
    }
    
    public static Shape append(Shape shape, long lastDimension) {
        long[] array = shape.asArray();
        long[] narray = new long[array.length + 1];
        System.arraycopy(array, 0, narray, 0, array.length);
        narray[array.length] = lastDimension;
        return Shape.of(narray);
    }
    /**
     * omputes sparse softmax cross entropy between `logits` and `labels`.
     * 
     * @param tf
     * @param labels `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
     * `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
     * must be an index in `[0, num_classes)`. Other values will raise an
     * exception when this op is run on CPU, and return `NaN` for corresponding
     * loss and gradient rows on GPU.
     * @param logits Per-label activations (typically a linear output) of shape
     * `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float16`, `float32`, or
     * `float64`. These activation energies are interpreted as unnormalized log
     * probabilities.
     * @return A `Tensor` of the same shape as `labels` and of the same type as `logits`
     * with the softmax cross entropy loss.
     */
    public static Operand sparse_softmax_cross_entropy_with_logits(Ops tf, Operand labels, Operand logits) {
        //assert shapeIsCompatible(labels.asOutput().shape(), logits.asOutput().shape()):
        //        String.format("Shapes %s and %s are incompatible", 
        //                labels.asOutput().shape(), logits.asOutput().shape());
        Operand precise_logits = logits;
        boolean convertToFloat32 = logits.asOutput().dataType() == TFloat16.DTYPE ||
                logits.asOutput().dataType() == TBfloat16.DTYPE;
        if(convertToFloat32 )
            precise_logits = tf.dtypes.cast(logits, TFloat32.DTYPE);
        Shape labelsShape = labels.asOutput().shape();
        Shape logitsShape = logits.asOutput().shape();
        Operand labels_shape = tf.shape(labels);
        // TODO
        //Shape newLogitsShape = squeeze(logitsShape, -1);
        Shape logitsShortened = shorten(logitsShape, -1);
        boolean staticShapesFullyDefined = !labelsShape.hasUnknownDimension() &&
               ! logitsShape.hasUnknownDimension();
        if(logitsShape.numDimensions() == 0) {
            throw new IllegalArgumentException(
                    String.format("Logits cannot be scalars - received shape %s.", logitsShape));
        }
        if(staticShapesFullyDefined && !labelsShape.equals(logitsShortened)) {
            throw new IllegalArgumentException(
                    String.format("Shape mismatch: The shape of labels (received %s) " +
                       "should equal the shape of logits except for the last " +
                       "dimension (received %s).", 
                            labelsShape, logitsShape));
        }
        // Check if no reshapes are required.
        if(logitsShape.numDimensions() == 2) {
            SparseSoftmaxCrossEntropyWithLogits smax = tf.nn.sparseSoftmaxCrossEntropyWithLogits(precise_logits, labels);
            Operand loss = smax.loss();
            if(logits.asOutput().dataType() == TFloat16.DTYPE) {
                loss = tf.dtypes.cast(loss, TFloat16.DTYPE);
            }
            return loss;
        }
        
        List<Operand> shapeChecks = new ArrayList<>();
        //if(!staticShapesFullyDefined){
        //    Shape logitsD = squeeze()
            //tf.assertThat(tf.math.equal(tf.shape(labels),  tf.shape(logits)));
        //}
        
        // Reshape logits to 2 dim, labels to 1 dim.
        
        long numClassses = logitsShape.size(logitsShape.numDimensions()-1);
        
        precise_logits = tf.reshape(precise_logits,tf.constant(new long[] {-1, numClassses}));
        labels = tf.reshape(labels, tf.constant(-1));
        SparseSoftmaxCrossEntropyWithLogits smax = tf.nn.sparseSoftmaxCrossEntropyWithLogits(precise_logits, labels);
        Operand cost = smax.loss();
        cost = tf.reshape(cost, labels_shape);
        if(logits.asOutput().dataType() == TFloat16.DTYPE) {
                cost = tf.dtypes.cast(cost, TFloat16.DTYPE);
            }
        
        
        // TODO how to change this to Operand
        /***********************
        List<Op> updateOperations = new ArrayList<>();
        updateOperations.add(cost);
        Scope scope = tf.scope();
        scope = scope.withName("sparse_softmax_cross_entropy_with_logits");
        scope = scope.withControlDependencies(updateOperations);
        return NoOp.create(scope);
        * ***************/
        return cost;
    }
    
    
     /// END nn OPS
    
    //TODO for debug, remove when done
    
    private static void debug( String prefix, Operand operand) {
        LossesImpl.debug(prefix, operand);
    }

    
    
}
