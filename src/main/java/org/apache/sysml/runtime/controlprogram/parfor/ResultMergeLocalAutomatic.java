/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.controlprogram.parfor;


import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptimizerRuleBased;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;

public class ResultMergeLocalAutomatic extends ResultMerge
{
	private static final long serialVersionUID = 1600893100602101732L;
	
	private ResultMerge _rm = null;
	
	public ResultMergeLocalAutomatic( MatrixObject out, MatrixObject[] in, String outputFilename, boolean accum ) {
		super( out, in, outputFilename, accum );
	}

	@Override
	public MatrixObject executeSerialMerge() 
		throws DMLRuntimeException 
	{
		Timing time = new Timing(true);
		
		MatrixCharacteristics mc = _output.getMatrixCharacteristics();
		long rows = mc.getRows();
		long cols = mc.getCols();
		
		if( OptimizerRuleBased.isInMemoryResultMerge(rows, cols, OptimizerUtils.getLocalMemBudget()) )
			_rm = new ResultMergeLocalMemory( _output, _inputs, _outputFName, _isAccum );
		else
			_rm = new ResultMergeLocalFile( _output, _inputs, _outputFName, _isAccum );
		
		MatrixObject ret = _rm.executeSerialMerge();

		LOG.trace("Automatic result merge ("+_rm.getClass().getName()+") executed in "+time.stop()+"ms.");

		return ret;
	}
	
	@Override
	public MatrixObject executeParallelMerge(int par) 
		throws DMLRuntimeException 
	{
		MatrixCharacteristics mc = _output.getMatrixCharacteristics();
		long rows = mc.getRows();
		long cols = mc.getCols();
		
		if( OptimizerRuleBased.isInMemoryResultMerge(par * rows, cols, OptimizerUtils.getLocalMemBudget()) )
			_rm = new ResultMergeLocalMemory( _output, _inputs, _outputFName, _isAccum );
		else
			_rm = new ResultMergeLocalFile( _output, _inputs, _outputFName, _isAccum );
		
		return _rm.executeParallelMerge(par);
	}
}
