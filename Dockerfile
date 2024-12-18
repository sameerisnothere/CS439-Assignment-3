FROM bitnami/spark:latest

ENV IVY_HOME=/tmp/.ivy2
RUN mkdir -p $IVY_HOME



# Copy data and scripts into the container
COPY data /data
COPY scripts /scripts

# Set the working directory
WORKDIR /scripts

# Run the EDA script
ENTRYPOINT ["spark-submit", "eda_script.py"]
