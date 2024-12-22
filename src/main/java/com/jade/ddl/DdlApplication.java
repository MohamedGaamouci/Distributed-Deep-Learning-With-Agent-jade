package com.jade.ddl;

import jade.core.Profile;
import jade.core.ProfileImpl;
import jade.core.Runtime;
import jade.wrapper.AgentContainer;
import jade.wrapper.AgentController;
import jade.wrapper.StaleProxyException;

public class DdlApplication {

    public static void main(String[] args) {
        final int num_workers = 3;
        // Start the JADE runtime environment
        Runtime runtime = Runtime.instance();

        // Create the main container
        Profile mainProfile = new ProfileImpl();
        mainProfile.setParameter(Profile.MAIN, "true");
        mainProfile.setParameter(Profile.GUI, "true");
        AgentContainer mainContainer = runtime.createMainContainer(mainProfile);

        try {
            // Start the CoordinatorAgent
            AgentController coordinatorAgent = mainContainer.createNewAgent(
                    "Coordinator",
                    "com.jade.ddl.Agents.CoordinatorAgent",
                    new Object[]{num_workers}
            );
            coordinatorAgent.start();

            // Start the DataDistributorAgent
            AgentController dataDistributorAgent = mainContainer.createNewAgent(
                    "DataDistributor",
                    "com.jade.ddl.Agents.DataDistributorAgent",
                    null
            );
            dataDistributorAgent.start();

            // Start the ModelAggregatorAgent
            AgentController modelAggregatorAgent = mainContainer.createNewAgent(
                    "ModelAggregator",
                    "com.jade.ddl.Agents.ModelAggregatorAgent",
                    new Object[]{num_workers}
            );
            modelAggregatorAgent.start();

            // Start Worker agents
            for (int i = 1; i <= 3; i++) {
                AgentController workerAgent = mainContainer.createNewAgent(
                        "Worker" + i,
                        "com.jade.ddl.Agents.WorkerAgent",
                        null
                );
                workerAgent.start();
            }

            AgentController TestAgent = mainContainer.createNewAgent(
                    "Test",
                    "com.jade.ddl.Agents.TestAgent",
                    null
            );
            TestAgent.start();

            System.out.println("All agents have been successfully started!");

        } catch (StaleProxyException e) {
            e.printStackTrace();
        }
    }

}
