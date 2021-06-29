import React, { useState} from 'react';
import axios from 'axios';
import Avatar from '@material-ui/core/Avatar';
import Button from '@material-ui/core/Button';
import CssBaseline from '@material-ui/core/CssBaseline';
import TextField from '@material-ui/core/TextField';
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';
import PostAddIcon from '@material-ui/icons/PostAdd';
import Select from '@material-ui/core/Select';
import MenuItem from '@material-ui/core/MenuItem';
import InputLabel from '@material-ui/core/InputLabel';
import Box from '@material-ui/core/Box';
import Paper from '@material-ui/core/Paper';

import Navbar from '../components/Navbar';

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(8),
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  avatar: {
    margin: theme.spacing(1),
    backgroundColor: theme.palette.secondary.main,
  },
  form: {
    width: '100%', // Fix IE 11 issue.
    marginTop: theme.spacing(1),
  },
  submit: {
    margin: theme.spacing(3, 0, 2),
  },
  formControl: {
    margin: theme.spacing(1),
    minWidth: 120,
  },
}));

export default function EvalForm(props) {
  let datasets = props.datasets;
  let models = props.models;
  console.log(datasets);
  const classes = useStyles();
  const [openDS, setOpenDS] = React.useState(false);
  const [openModel, setOpenModel] = React.useState(false);
  const [modelID, setModelID] = React.useState(0);
  const [datasetID, setDatasetID] = React.useState(0);

  const [values, setValues] = useState({
        name: 'test_eval',
        model_type: 'regression',
        model_id: 0,
        dataset_id: 0,
        metadata: {},
  });

  const handleDropdownChangeModelID = (event) => {
    setModelID(event.target.value);
    setValues({ ...values, 'model_id': event.target.value });
  };

  const handleDropdownChangeDatasetID = (event) => {
    setDatasetID(event.target.value);
    setValues({ ...values, 'dataset_id': event.target.value });
  };

  const handleModelClose = () => {
    setOpenModel(false);
  };

  const handleModelOpen = () => {
    setOpenModel(true);
  };

  const handleDatasetClose = () => {
    setOpenDS(false);
  };

  const handleDatasetOpen = () => {
    setOpenDS(true);
  };


    const handleSubmit = async (e) => {
        e.preventDefault();
        const { name, model_id, dataset_id, metadata } = values;
        const payload = { name, model_id, dataset_id, metadata };

        await axios.post('/modelEvaluations', payload).then(() => {window.location="/";});
    };

    const handleChange = name => e => {
        setValues({ ...values, [name]: e.target.value });
    };

    const handleCancel = async() => {
      window.location="/";
    };

  return (
    <>
      <Navbar />
          <Box ml={36} mr={36} pr={15} pl={15}>
            <Paper elevation={5}>
              <Box m={2} pl={3} pr={3} pb={3}>
          <CssBaseline />
          <div className={classes.paper}>
            <Avatar className={classes.avatar}>
              <PostAddIcon />
            </Avatar>
            <Typography component="h1" variant="h5">
              Add Evaluation
            </Typography>
            <form className={classes.form} onSubmit={handleSubmit}>
              <TextField
                variant="outlined"
                margin="normal"
                required
                fullWidth
                id="name"
                label="Evaluation Name"
                name="name"
                autoComplete="Evaluation Name"
                autoFocus
                onChange={handleChange('name')}
              />

              <div className={classes.formControl}>
                <InputLabel id="dataset_id">Dataset</InputLabel>
                <Select
                  labelId="dataset_id"
                  id="did"
                  open={openDS}
                  fullWidth
                  onClose={handleDatasetClose}
                  onOpen={handleDatasetOpen}
                  value={datasetID}
                  onChange={handleDropdownChangeDatasetID}
                >
                  {datasets.map((dataset) =>
                    <MenuItem value={dataset.dataset_id}>{dataset.name}</MenuItem>
                  )}
                </Select>
              </div>

              <div className={classes.formControl}>
                <InputLabel id="model_id">Model</InputLabel>
                <Select
                  labelId="model_id"
                  id="mid"
                  fullWidth
                  open={openModel}
                  onClose={handleModelClose}
                  onOpen={handleModelOpen}
                  value={modelID}
                  onChange={handleDropdownChangeModelID}
                >
                  {models.map((model) =>
                    <MenuItem value={model.model_id}>{model.name}</MenuItem>
                  )}
                </Select>
              </div>

              <TextField
                variant="filled"
                margin="normal"
                fullWidth
                name="description"
                label="Description"
                id="description"
                autoComplete="Evaluation Description"
              />

              <div className="row">

                  <div className="col">
                    <Button
                      type="submit"
                      fullWidth
                      variant="contained"
                      color="secondary"
                      size="large"
                      className={classes.submit}
                      onClick={handleSubmit}
                    >
                      Add
                    </Button>
                  </div>

                  <div className="col">
                    <Button
                      fullWidth
                      variant="contained"
                      color="primary"
                      size="large"
                      className={classes.submit}
                      onClick={handleCancel}
                    >
                      Cancel
                    </Button>
                  </div>

                </div>
              </form>
            </div>
            </Box>
            </Paper>
            </Box>
    </>
  );
};